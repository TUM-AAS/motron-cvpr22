from typing import Tuple

import torch
import numpy as np
from scipy.stats import gaussian_kde
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


def compute_kde_nll(y, y_pred):
    bs, sp, ts, ns, d = y_pred.shape
    kde_ll = torch.zeros((bs, ts, ns))

    for b in range(bs):
        for t in range(ts):
            for n in range(ns):
                try:
                    kde = gaussian_kde(y_pred[b, :, t, n].T)
                    pdf = kde.logpdf(y[b, t, n].T)
                    kde_ll[b, t, n] = torch.tensor(pdf)
                except np.linalg.LinAlgError:
                    print(b, t, n)
                    print('nan')
                    pass

    return -kde_ll


class NegativeLogLikelihood(Metric):
    def __init__(self, output_transform=lambda x: x, keep_time_dim: bool = True):
        self.y = None
        self.y_pred = None
        self.keep_time_dim = keep_time_dim
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
            raise NotComputableError('MeanPerJointPositionError must have at least one example before it can be computed.')
        ret = compute_kde_nll(self.y, self.y_pred)
        if not self.keep_time_dim:
            ret = ret.sum(-1).mean()
        return ret
