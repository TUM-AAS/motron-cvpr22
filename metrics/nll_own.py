from typing import Tuple

import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


def compute_nll(y, y_pred):
    ll = ((y_pred.log_prob(y))
          .mean(-1))
    return -ll


class NegativeLogLikelihoodOwn(Metric):
    def __init__(self, output_transform=lambda x: x):
        self.nll = None
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.nll = None
        super().reset()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        if len(output) == 2:
            y_pred, y = output
        else:
            y_pred, y, _ = output

        if self.nll is None:
            self.nll = compute_nll(y, y_pred)
        else:
            self.nll = torch.cat([self.nll, compute_nll(y, y_pred)], dim=0)

    def compute(self):
        if self.nll is None:
            raise NotComputableError('MeanPerJointPositionError must have at least one example before it can be computed.')
        return self.nll.mean(dim=0)
