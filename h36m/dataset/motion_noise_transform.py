import torch

from motion.quaternion import Quaternion
from motion.bingham import Bingham


class MotionNoiseTransform(torch.nn.Module):
    def __init__(self, z):
        super().__init__()
        self.bingham = Bingham(
            M=torch.tensor([[0, 1, 0, 0.],
                          [0, 0, 1, 0.],
                          [0, 0, 0, 1.],
                          [1, 0, 0, 0.]]).float(),
            Z=torch.tensor([-z, -z, -z, 0]).float()
        )

    def forward(self, q):
        ts, ns, ds = q.shape
        q_n = self.bingham.sample((1, ns))
        return Quaternion.mul_(q, q_n)
