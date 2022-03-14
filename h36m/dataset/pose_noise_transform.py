import torch

from motion.quaternion import Quaternion
from motion.bingham import Bingham


class PoseNoiseTransform(torch.nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, q):
        ts, ns, ds = q.shape
        q_dot = Quaternion.mul_(q[1:], Quaternion.conjugate_(q[:-1]))
        q_samp = ((torch.rand((ts-1, ns, ds)) - 0.5) * torch.tensor([1., self.std, self.std, self.std]))
        q_samp = q_samp / q_samp.norm(dim=-1, keepdim=True)
        q_dot = Quaternion.mul_(q_dot, q_samp)

        q_i = q[0]
        q_out = [q_i]
        for i in range(ts-1):
            q_i = Quaternion.mul_(q_dot[i], q_i)
            q_out.append(q_i)
        return torch.stack(q_out, dim=0)
