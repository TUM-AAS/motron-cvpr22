from typing import Tuple

import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

from motion import Quaternion


class QuaternionAngle(Metric):
    def __init__(self, output_transform=lambda x: x, ignore_root=True):
        self.y = None
        self.y_pred = None
        self.ignore_root = ignore_root
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.y = None
        self.y_pred = None
        super().reset()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        if len(output) == 2:
            y_pred, y = output
        else:
            y_pred, y, _ = output

        if self.y is None:
            self.y = y
            self.y_pred = y_pred
        else:
            self.y = torch.cat([self.y, y], dim=0)
            self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)

    def compute(self):
        if self.y is None:
            raise NotComputableError('MeanAngleError must have at least one example before it can be computed.')
        error = (Quaternion(self.y) * Quaternion(self.y_pred).conjugate).angle
        if self.ignore_root:
            return error[..., 1:]
        return error
