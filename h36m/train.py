import os

# There is a bug with jit in torch 1.8 on GPU
# TODO remove on upgrade to 1.9

os.environ['PYTORCH_JIT'] = '0'
import sys

sys.path.append('./Motion/')
import os
import yaml
from typing import Sequence

import numpy as np

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from ignite.contrib.handlers import ProgressBar, CosineAnnealingScheduler
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage, Loss, Average, BatchWise
from ignite.contrib.handlers.tensorboard_logger import *


from misc.helper import remove_static_nodes, p_q_mode_output_transform, position_mode_output_transform, \
    default_output_transform, motion_variance

from motion.utils.torch import mutual_inf_px
from motion.utils.ignite import set_default_tb_train_logging
from motion.utils.torch import set_seed
from h36m.dataset.h36m_dataset import H36MDataset
from h36m.dataset.h36m_test_dataset import H36MTestDataset
from h36m.dataset.h36m_torch_dataset import H36MTorchDataset
from metrics import MeanPerJointPositionError, MeanAngleL2Error, MeanAngleError, MeanSquaredError,\
    NegativeLogLikelihoodOwn
from motion import Motion, Quaternion
from h36m.skeleton import H36MSkeleton


def train(output_log_path: str,
          lr: float,
          batch_size: int,
          num_epochs: int,
          eval_frequency: int = None,
          random_prediction_horizon: bool = False,
          curriculum_it: int = 0,
          batch_size_eval: int = 0,
          num_iteration_eval: int = 0,
          clip_grad_norm: float = None,
          device: str = 'cpu',
          seed: int = 52345,
          num_workers: int = 0,
          detect_anomaly: bool = False,
          **kwargs) -> None:
    """
    Trains Motion on H3.6M Dataset
    """
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # Init seed
    set_seed(seed)

    # Load skeleton configuration
    with open(kwargs['graph_config'], 'r') as stream:
        skeleton = H36MSkeleton(**yaml.safe_load(stream))

    # Create model
    model = Motion(skeleton,
                   position=False,
                   T=skeleton.nodes_type_id_dynamic,
                   **kwargs).to(device)

    print(f"Created Model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # Load the H3.6M Dataset from disk
    h36m_dataset = H36MDataset(**kwargs)

    prediction_horizon_train = kwargs['prediction_horizon_train']
    prediction_horizon_eval = kwargs['prediction_horizon_eval']

    # Create TorchDatasets for training and validation
    dataset_train = H36MTorchDataset(h36m_dataset, subjects=kwargs['subjects_train'],
                                     **kwargs)
    dataset_train.mirror(skeleton)  # Dataset Augmentation mirroring left and right

    data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=True)

    dataset_eval = H36MTorchDataset(h36m_dataset, subjects=kwargs['subjects_eval'], **kwargs)
    eval_idx = list(range(0, len(dataset_eval), 100))  # Evaluate on every 100 data points
    eval_set = torch.utils.data.Subset(dataset_eval, eval_idx)
    data_loader_eval = DataLoader(eval_set, shuffle=False, batch_size=batch_size_eval, num_workers=0)

    dataset_train_eval = H36MTorchDataset(h36m_dataset, subjects=kwargs['subjects_train'], **kwargs)
    data_loader_train_eval = DataLoader(dataset_train_eval, shuffle=True, batch_size=batch_size_eval, num_workers=0)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

    class CurriculumLearning:
        def __init__(self):
            self.param_groups = [{'curriculum_factor': 0.}]

    curriculum = CurriculumLearning()

    if curriculum_it is not None and curriculum_it > 0.0:
        curriculum_scheduler = CosineAnnealingScheduler(curriculum,
                                                        'curriculum_factor',
                                                        start_value=1.0,
                                                        end_value=0.,
                                                        cycle_size=curriculum_it,
                                                        start_value_mult=0.0
                                                        )

    # Define pre-process function to transform input to correct data type
    def preprocess(engine: Engine):
        engine.state.batch = [Quaternion.qfix_positive_(remove_static_nodes(t.to(device), skeleton.dynamic_nodes))
                              for t in engine.state.batch[:]] \
                             + [t.to(device) for t in engine.state.batch[:]]

    # Define process function which is called during every training step
    def train_step(engine: Engine, batch: Sequence[torch.Tensor]):
        model.train()
        optimizer.zero_grad()
        x, y, _, _ = batch

        ph = max(int(np.rint((1. - curriculum.param_groups[0]['curriculum_factor']) * prediction_horizon_train)), 1)
        if ph > 1 and random_prediction_horizon:
            ph = np.random.randint(1, ph)
        y = y[:, :ph]

        p_q, _, _, kwargs = model(x, None, ph, None)

        loss_unscaled = model.loss(p_q, y)

        loss = 1e-3 * loss_unscaled

        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        return loss_unscaled, p_q, y

    def validation_step(engine: Engine, batch: Sequence[torch.Tensor]):
        model.eval()
        with torch.no_grad():
            x, y, x_org, y_org = batch
            model_out, _, _, _ = model(x, None, prediction_horizon_eval)
            return model_out, y, y_org

    # Define ignite metrics
    mse = MeanSquaredError(output_transform=p_q_mode_output_transform(skeleton))
    mae = MeanAngleError(output_transform=p_q_mode_output_transform(skeleton))
    mae_l2 = MeanAngleL2Error(output_transform=p_q_mode_output_transform(skeleton))
    mpjpe_t = MeanPerJointPositionError(output_transform=position_mode_output_transform(skeleton))
    mpjpe = MeanPerJointPositionError(output_transform=position_mode_output_transform(skeleton), keep_time_dim=False)
    nll_metric = NegativeLogLikelihoodOwn(output_transform=default_output_transform(skeleton))
    loss_metric = Loss(model.loss, output_transform=default_output_transform(skeleton))
    mi_metric = Average(output_transform=lambda x: mutual_inf_px(x[1].mixture_distribution))
    latent_max = Average(output_transform=lambda x: x[1].mixture_distribution.probs.amax(dim=-1).mean())
    motion_variance_metric = Average(output_transform=lambda x: motion_variance(x[1].component_distribution.mean))
    # Define Training, Evaluation and Test Engines and attach metrics
    trainer = Engine(train_step)
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'Loss')
    mi_metric.attach(trainer, 'MI_p_zx', BatchWise())
    latent_max.attach(trainer, 'Latent_Max', BatchWise())
    motion_variance_metric.attach(trainer, 'Motion_var', BatchWise())
    trainer.add_event_handler(Events.ITERATION_STARTED, preprocess)
    if curriculum_it is not None and curriculum_it > 0.0:
        trainer.add_event_handler(Events.ITERATION_STARTED, curriculum_scheduler)

    evaluator = Engine(validation_step)
    evaluator.add_event_handler(Events.ITERATION_STARTED, preprocess)
    loss_metric.attach(evaluator, 'Loss')
    mpjpe.attach(evaluator, 'MPJPE')
    mpjpe_t.attach(evaluator, 'MPJPE_T')
    mae.attach(evaluator, 'MAE')
    mae_l2.attach(evaluator, 'MAE_L2')
    mse.attach(evaluator, 'MSE')

    # Evaluate on Train dataset
    evaluator_train = Engine(validation_step)
    evaluator_train.add_event_handler(Events.ITERATION_STARTED, preprocess)
    loss_metric.attach(evaluator_train, 'Loss')
    mpjpe.attach(evaluator_train, 'MPJPE')
    mpjpe_t.attach(evaluator_train, 'MPJPE_T')
    mae.attach(evaluator_train, 'MAE')
    mae_l2.attach(evaluator_train, 'MAE_L2')
    mse.attach(evaluator_train, 'MSE')

    testers = dict()
    if kwargs['dataset_downsample_factor'] > 1:
        for action in kwargs['test_actions'] + ['average']:
            testers[action] = Engine(validation_step)
            mpjpe.attach(testers[action], 'MPJPE')
            mpjpe_t.attach(testers[action], 'MPJPE_T')
            mae.attach(testers[action], 'MAE')
            mae_l2.attach(testers[action], 'MAE_L2')
            mse.attach(testers[action], 'MSE')
            nll_metric.attach(testers[action], 'NLL')
            loss_metric.attach(testers[action], 'Loss')
            testers[action].add_event_handler(Events.ITERATION_STARTED, preprocess)

    # Setup tensorboard logging and progressbar for training
    tb_logger = TensorboardLogger(log_dir=os.path.join(output_log_path, 'tb'))
    setup_logging(tb_logger, trainer, evaluator, evaluator_train, optimizer, model, testers, eval_frequency)

    pbar = ProgressBar()
    pbar.attach(trainer, ['Loss']),
    pbar.attach(evaluator),
    pbar.attach(evaluator_train)

    # Setup evaluation process between epochs
    if eval_frequency is not None:
        @trainer.on(Events.ITERATION_COMPLETED(every=eval_frequency))
        def train_epoch_completed(engine: Engine):
            for action, tester in testers.items():
                dataset_test_action = H36MTestDataset(h36m_dataset,
                                                      action=action,
                                                      subjects=kwargs['subjects_test'],
                                                      num_samples=256,
                                                      **kwargs)
                tester.run(DataLoader(dataset_test_action, batch_size=256))

            # Run evaluation on random chunk of dataset
            evaluator.run(data_loader_eval)

            evaluator_train.run(data_loader_train_eval, epoch_length=num_iteration_eval)

            torch.save(model.state_dict(), os.path.join(output_log_path, f"checkpoints/{engine.state.iteration}.pth.tar"))

    trainer.run(data_loader_train, max_epochs=num_epochs)

    tb_logger.close()


def setup_logging(tb_logger: TensorboardLogger, trainer: Engine, evaluator: Engine, evaluator_train, optimizer: Optimizer,
                  model: torch.nn.Module, testers: dict = None, eval_frequency=1):
    set_default_tb_train_logging(tb_logger, trainer, optimizer, model)

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        **{'tag': "training",
           'metric_names': ["Latent_Max", "MI_p_zx", "Motion_var"],
           }
    )

    tb_custom_scalar_layout = {}

    # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
    # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
    # `trainer` instead of `evaluator`.
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.COMPLETED,
        **{'tag': "validation",
           'metric_names': ["Loss", "MPJPE", "MPJPE_T", "MAE", "MAE_L2", "MSE"],
           'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
           }
    )

    tb_custom_scalar_layout = {
        **tb_custom_scalar_layout,
        **{
            'Validation Metrics': {
                'MAE': ['Multiline', [rf"validation/MAE/.*"]],
                'MAE_L2': ['Multiline', [rf"validation/MAE_L2/.*"]],
                'MPJPE_T': ['Multiline', [rf"validation/MPJPE_T/.*"]]
            }
        }
    }

    tb_logger.attach_output_handler(
        evaluator_train,
        event_name=Events.COMPLETED,
        **{'tag': "validation_train",
           'metric_names': ["Loss", "MPJPE", "MPJPE_T", "MAE", "MAE_L2", "MSE"],
           'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
           }
    )

    tb_custom_scalar_layout = {
        **tb_custom_scalar_layout,
        **{
            'Validation Metrics': {
                'MAE': ['Multiline', [rf"validation_train/MAE/.*"]],
                'MAE_L2': ['Multiline', [rf"validation_train/MAE_L2/.*"]],
                'MPJPE_T': ['Multiline', [rf"validation_train/MPJPE_T/.*"]]
            }
        }
    }

    if testers is not None:
        tb_cs_layout_mae = {}
        tb_cs_layout_mae_l2 = {}
        tb_cs_layout_mpjpe_t = {}
        for action, tester in testers.items():
            tb_logger.attach_output_handler(
                tester,
                event_name=Events.COMPLETED,
                **{'tag': f"test/{action}",
                   'metric_names': ["Loss", "MAE", "MAE_L2", "MPJPE_T", "MSE", "NLL"],
                   'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
                   }
            )
            tb_cs_layout_mae[f"{action}_MAE"] = ['Multiline', [rf"test/{action}/MAE/.*"]]
            tb_cs_layout_mae_l2[f"{action}_MAE_L2"] = ['Multiline', [rf"test/{action}/MAE_L2/.*"]]
            tb_cs_layout_mpjpe_t[f"{action}_MPJPE_T"] = ['Multiline', [rf"test/{action}/MPJPE_T/.*"]]

        tb_custom_scalar_layout = {
            **tb_custom_scalar_layout,
            **{'Test Actions MAE': tb_cs_layout_mae,
               'Test Actions MAE L2': tb_cs_layout_mae_l2,
               'Test Actions MPJPE_T': tb_cs_layout_mpjpe_t
               }
        }
    tb_logger.writer.add_custom_scalars(tb_custom_scalar_layout)


import argparse
import shutil
import datetime

from motion.utils.os import maybe_makedir

parser = argparse.ArgumentParser(description='H3.6M Download')
parser.add_argument('--config',
                    type=str,
                    help='The config file for training',
                    default='./config/h36m.yaml')

parser.add_argument('--info',
                    type=str,
                    help='Additional information which will be added to the output path',
                    default='')

parser.add_argument('--debug',
                    type=bool,
                    help='Debug Flag. No atrifacts will be saved to disk.',
                    default=False)

parser.add_argument('--device',
                    type=str,
                    help='Training Device.',
                    default='cpu')

args = parser.parse_args()

if __name__ == '__main__':
    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)
    params['info'] = args.info
    params['debug'] = args.debug
    params['device'] = args.device

    if not params['debug']:
        dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not params['info'] == '':
            info_str = f"_{params['info']}"
        else:
            info_str = ''
        params['output_log_path'] = os.path.join(params['output_path'], f"{dt_str}{info_str}")
        maybe_makedir(os.path.join(params['output_log_path'], 'checkpoints'))
        maybe_makedir(params['output_log_path'])
        with open(os.path.join(params['output_log_path'], 'config.yaml'), 'w') as config_file:
            yaml.dump(params, config_file)
        shutil.copytree('./Motion/', os.path.join(params['output_log_path'], 'Motion'))
    train(**params)

