import sys

import zarr

sys.path.append('./Motion/')
import os

# There is a bug with jit in torch 1.8 on GPU
# TODO remove on upgrade to 1.9
os.environ['PYTORCH_JIT'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1s'

import yaml
from typing import Sequence, List

import torch
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset
from ignite.contrib.handlers import ProgressBar, CosineAnnealingScheduler, create_lr_scheduler_with_warmup, LRScheduler
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage, Loss, Average, BatchWise
from ignite.contrib.handlers.tensorboard_logger import *


from misc.helper import remove_static_nodes, p_q_mode_output_transform, position_mode_output_transform, \
    default_output_transform, motion_variance

from amass.amass_torch_dataset import AMASSTorchDataset
from amass.skeleton import AMASSSkeleton

from motion.utils.torch import mutual_inf_px
from motion.utils.ignite import set_default_tb_train_logging
from motion.utils.torch import to_dtype, set_seed
from metrics import MeanPerJointPositionError, MeanAngleL2Error, MeanAngleError, MeanOriginPositionError
from motion import Motion, Quaternion


def train(output_log_path: str,
          dataset_path: str,
          datasets_train: List,
          datasets_test: List,
          lr: float,
          batch_size: int,
          num_epochs: int,
          num_iteration_per_epoch: int = None,
          eval_frequency: int = None,
          random_prediction_horizon: bool = False,
          curriculum_it: int = 0,
          batch_size_eval: int = 0,
          num_iteration_eval: int = 0,
          clip_grad_norm: float = None,
          device: str = 'cpu',
          seed: int = 12345,
          num_workers: int = 0,
          detect_anomaly: bool = False,
          **kwargs) -> None:
    """
    Trains Motion on H3.6M Dataset
    """
    torch.autograd.set_detect_anomaly(False)

    # Init seed
    set_seed(seed)

    # Load skeleton configuration
    with open(kwargs['graph_config'], 'r') as stream:
        skeleton = AMASSSkeleton(**yaml.safe_load(stream))

    # Create model
    model = Motion(skeleton,
                   position=False,
                   T=skeleton.nodes_type_id_dynamic,
                   **kwargs).to(device)
    print(f"Created Model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    prediction_horizon = kwargs['prediction_horizon']

    # Create TorchDatasets for training and validation
    train_datasets = []
    eval_datasets = []
    for dataset in datasets_train:
        z_index = zarr.open(os.path.join(dataset_path, dataset, 'poses_index.zarr'), 'r')
        z_poses = zarr.open(os.path.join(dataset_path, dataset, 'poses.zarr'), 'r')
        z_trans = zarr.open(os.path.join(dataset_path, dataset, 'trans.zarr'), 'r')
        train_val_split = int(len(z_index) * 0.95)
        train_datasets.append(AMASSTorchDataset(z_index[:train_val_split], z_poses, trans_data=None,  **kwargs))
        train_datasets[-1].dataset_name = dataset
        eval_datasets.append(AMASSTorchDataset(z_index[train_val_split:], z_poses, trans_data=None, **kwargs))
    dataset_train = ConcatDataset(train_datasets)
    print(f"Train dataset: {len(dataset_train)}")
    data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    dataset_eval = ConcatDataset(eval_datasets)
    print(f"Eval dataset: {len(dataset_eval)}")
    eval_idx = list(torch.linspace(0, len(dataset_eval) - 1, 3000, dtype=torch.int).numpy())
    eval_set = torch.utils.data.Subset(dataset_eval, eval_idx)
    data_loader_eval = DataLoader(eval_set, shuffle=False, batch_size=batch_size_eval, num_workers=0)

    eval_idx = list(torch.linspace(0, len(dataset_train) - 1, 1000, dtype=torch.int).numpy())
    eval_train_set = torch.utils.data.Subset(dataset_train, eval_idx)
    data_loader_train_eval = DataLoader(eval_train_set, shuffle=False, batch_size=batch_size_eval, num_workers=0)

    test_datasets = []
    for dataset in datasets_test:
        z_index = zarr.open(os.path.join(dataset_path, dataset, 'poses_index.zarr'), 'r')
        z_poses = zarr.open(os.path.join(dataset_path, dataset, 'poses.zarr'), 'r')
        z_trans = zarr.open(os.path.join(dataset_path, dataset, 'trans.zarr'), 'r')
        test_datasets.append(AMASSTorchDataset(z_index, z_poses, trans_data=None, **kwargs))
    dataset_test = ConcatDataset(test_datasets)
    print(f"Test dataset: {len(dataset_test)}")
    test_idx = list(torch.linspace(0, len(dataset_test)-1, 5000, dtype=torch.int).numpy())
    test_set = torch.utils.data.Subset(dataset_test, test_idx)
    data_loader_test = DataLoader(test_set, shuffle=False, batch_size=batch_size_eval, num_workers=0)

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

        ph = max(int(np.rint((1. - curriculum.param_groups[0]['curriculum_factor']) * prediction_horizon)), 1)
        if ph > 1 and random_prediction_horizon:
            ph = np.random.randint(1, ph)
        y = y[:, :ph]

        p_q, _, _, kwargs = model(x, None, ph, None)

        loss_unscaled = model.loss(p_q, y)

        loss = 1e-3 * loss_unscaled
        #loss = torch.mean(torch.sum(torch.abs(p_q.mode - y), dim=2))

        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        return loss_unscaled, p_q, y

    def validation_step(engine: Engine, batch: Sequence[torch.Tensor]):
        model.eval()
        with torch.no_grad():
            x_q, y_q, _, _ = batch
            model_out = model(q=x_q, ph=prediction_horizon)
            return model_out, y_q

    # Define ignite metrics
    def to_joint_pos(q):
        bs, ts, ns, ds = q.shape
        return skeleton(q.reshape(-1, ns, 4)).reshape(-1, ts, ns, 3)

    mae = MeanAngleError(output_transform=lambda out: (out[0][0].weighted_mean, out[1]), keep_time_dim=False)
    mae_t = MeanAngleError(output_transform=lambda out: (out[0][0].weighted_mean, out[1]))
    mae_l2 = MeanAngleL2Error(output_transform=lambda out: (out[0][0].weighted_mean, out[1]), keep_time_dim=False)
    mae_l2_t = MeanAngleL2Error(output_transform=lambda out: (out[0][0].weighted_mean, out[1]))
    #mpjpe_t = MeanPerJointPositionError(output_transform=lambda out: (to_joint_pos(out[0][0].mode), to_joint_pos(out[1])))
    #mpjpe = MeanPerJointPositionError(output_transform=lambda out: (to_joint_pos(out[0][0].mode), to_joint_pos(out[1])),
    #                                  keep_time_dim=False)
    #mope_t = MeanOriginPositionError(output_transform=lambda out: (out[0][1].mode, out[2]))
    #mope = MeanOriginPositionError(output_transform=lambda out: (out[0][1].mode, out[2]), keep_time_dim=False)
    loss_metric = Loss(model.loss, output_transform=lambda out: (out[0][0], out[1]))

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

    evaluator = Engine(validation_step)
    evaluator.add_event_handler(Events.ITERATION_STARTED, preprocess)
    loss_metric.attach(evaluator, 'Loss')
    #mpjpe.attach(evaluator, 'MPJPE')
    #mpjpe_t.attach(evaluator, 'MPJPE_T')
    mae.attach(evaluator, 'MAE')
    mae_t.attach(evaluator, 'MAE_T')
    mae_l2.attach(evaluator, 'MAE_L2')
    mae_l2_t.attach(evaluator, 'MAE_L2_T')
    #mope_t.attach(evaluator, 'MOPE_T')
    #mope.attach(evaluator, 'MOPE')

    tester = Engine(validation_step)
    tester.add_event_handler(Events.ITERATION_STARTED, preprocess)
    loss_metric.attach(tester, 'Loss')
    #mpjpe.attach(tester, 'MPJPE')
    #mpjpe_t.attach(tester, 'MPJPE_T')
    mae.attach(tester, 'MAE')
    mae_t.attach(tester, 'MAE_T')
    mae_l2.attach(tester, 'MAE_L2')
    mae_l2_t.attach(tester, 'MAE_L2_T')
    #mope_t.attach(tester, 'MOPE_T')
    #mope.attach(tester, 'MOPE')

    # Evaluate on Train dataset
    evaluator_train = Engine(validation_step)
    evaluator_train.add_event_handler(Events.ITERATION_STARTED, preprocess)
    loss_metric.attach(evaluator_train, 'Loss')
    #mpjpe.attach(evaluator_train, 'MPJPE')
    #mpjpe_t.attach(evaluator_train, 'MPJPE_T')
    mae.attach(evaluator_train, 'MAE')
    mae_t.attach(evaluator_train, 'MAE_T')
    mae_l2.attach(evaluator_train, 'MAE_L2')
    mae_l2_t.attach(evaluator_train, 'MAE_L2_T')
    #mope_t.attach(evaluator_train, 'MOPE_T')
    #mope.attach(evaluator_train, 'MOPE')

    # Setup tensorboard logging and progressbar for training
    tb_logger = TensorboardLogger(log_dir=os.path.join(output_log_path, 'tb'))
    setup_logging(tb_logger, trainer, evaluator, evaluator_train, optimizer, model, tester, eval_frequency)

    pbar = ProgressBar()
    pbar.attach(trainer, ['Loss']),
    pbar.attach(evaluator),
    pbar.attach(evaluator_train)


    # Setup evaluation process between epochs
    if eval_frequency is not None:
        @trainer.on(Events.ITERATION_COMPLETED(every=eval_frequency))
        def train_epoch_completed(engine: Engine):
            # Run evaluation on random chunk of dataset
            evaluator.run(data_loader_eval)

            torch.save(model.state_dict(),
                       os.path.join(output_log_path, f"checkpoints/{engine.state.iteration}.pth.tar"))

            tester.run(data_loader_test)

            evaluator_train.run(data_loader_train_eval)

    trainer.run(data_loader_train, max_epochs=num_epochs)

    # We need to close the logger with we are done
    tb_logger.close()


def setup_logging(tb_logger: TensorboardLogger, trainer: Engine, evaluator: Engine, evaluator_train, optimizer: Optimizer,
                  model: torch.nn.Module, tester = None, eval_frequency=1):
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
           'metric_names': ["Loss", "MAE", "MAE_L2", "MAE_T", "MAE_L2_T"],
           'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
           }
    )

    tb_custom_scalar_layout = {
        **tb_custom_scalar_layout,
        **{
            'Validation Metrics': {
                'MAE': ['Multiline', [rf"validation/MAE/.*"]],
                'MAE_L2': ['Multiline', [rf"validation/MAE_L2/.*"]]
            }
        }
    }

    tb_logger.attach_output_handler(
        evaluator_train,
        event_name=Events.COMPLETED,
        **{'tag': "validation_train",
           'metric_names': ["Loss", "MAE", "MAE_L2", "MAE_T", "MAE_L2_T"],
           'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
           }
    )

    tb_custom_scalar_layout = {
        **tb_custom_scalar_layout,
        **{
            'Validation Metrics': {
                'MAE': ['Multiline', [rf"validation_train/MAE/.*"]],
                'MAE_L2': ['Multiline', [rf"validation_train/MAE_L2/.*"]],
            }
        }
    }

    if tester is not None:
        tb_logger.attach_output_handler(
            tester,
            event_name=Events.COMPLETED,
            **{'tag': "test",
               'metric_names': ["Loss", "MAE", "MAE_L2", "MAE_T", "MAE_L2_T"],
               'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
               }
        )


import argparse
import shutil
import datetime

from motion.utils.os import maybe_makedir

parser = argparse.ArgumentParser(description='H3.6M Download')
parser.add_argument('--config',
                    type=str,
                    help='The config file for training',
                    default='./config/amass.yaml')

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

