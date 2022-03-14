from typing import Tuple

import numpy as np
import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

from motion import Quaternion


def mae_qq(q1: torch.Tensor, q2: torch.Tensor):
    """
    Mean angle error between two quaternion tensors.
    :param q1: First quaternions tensor.
    :param q2: Second quaternions tensor.
    :return: MAE
    """
    assert len(q1.shape) == 4 and len(q2.shape) == 4
    assert q1.shape[-1] == 4 and q2.shape[-1] == 4
    euler1 = Quaternion.euler_angle_(q1.contiguous(), 'zyx')
    euler2 = Quaternion.euler_angle_(q2.contiguous(), 'zyx')
    return mae_ee(euler1, euler2)


def mae_ee(e1: torch.Tensor, e2: torch.Tensor):
    """
    Mean angle error between two euler angle tensors.
    :param e1: First euler angles tensor.
    :param e2: Second euler angles tensor.
    :return: MAE
    """
    assert len(e1.shape) == 4 and len(e2.shape) == 4
    assert e1.shape[-1] == 3 and e2.shape[-1] == 3
    diff = torch.remainder(e2 - e1 + np.pi, 2 * np.pi) - np.pi
    return torch.mean(torch.sqrt(torch.sum(torch.square(diff), dim=-1)), dim=[0, -1])


class MeanAngleError(Metric):
    def __init__(self, output_transform=lambda x: x, ignore_root=True, keep_time_dim: bool = True):
        self.y = None
        self.y_pred = None
        self.ignore_root = ignore_root
        self.keep_time_dim = keep_time_dim
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.y = None
        self.y_pred = None
        super().reset()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        return
        if len(output) == 2:
            y_pred, y = output
        else:
            y_pred, y, _ = output
        y_pred = y_pred.clone()
        y = y.clone()

        if self.y is None:
            self.y = y
            self.y_pred = y_pred
        else:
            self.y = torch.cat([self.y, y], dim=0)
            self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)

    def compute(self):
        return 0.
        if self.y is None:
            raise NotComputableError('MeanAngleError must have at least one example before it can be computed.')
        if self.ignore_root:
            ret = mae_qq(self.y[..., 1:, :].clone(), self.y_pred[..., 1:, :].clone())
        else:
            ret = mae_qq(self.y.clone(), self.y_pred.clone())

        if not self.keep_time_dim:
            ret = ret.mean()

        return ret
