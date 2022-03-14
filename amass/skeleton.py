from typing import List

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from motion import Skeleton as MotionSkeleton, Quaternion


class AMASSSkeleton(MotionSkeleton):
    def __init__(self,
                 body_model_path: str,
                 parents: List,
                 joints_static: List,
                 joints_type: List):
        super().__init__()
        self._body_model_path = body_model_path
        self._parents = torch.tensor(parents)
        self._joints_static = torch.tensor(joints_static)
        self._joints_type = np.array(joints_type)
        self._joints_dynamic = torch.tensor([i for i in range(len(self._parents)) if i not in self._joints_static])
        self._offsets = BodyModel(self._body_model_path)().Jtr[0, :22].detach()
        self._compute_metadata()

    @property
    def num_joints(self):
        return len(self._joints_dynamic)

    @property
    def num_nodes(self):
        return len(self._parents)

    @property
    def num_dynamic_nodes(self):
        return len(self._joints_dynamic)

    @property
    def adjacency_matrix(self):
        return self._adjacency_matrix

    @property
    def chain_list(self):
        return self._chain_list

    @property
    def dynamic_nodes(self):
        return self._joints_dynamic

    @property
    def static_nodes(self):
        return self._joints_static

    @property
    def nodes_type_id_dynamic(self):
        joint_id_string_wo = []
        for joint_id_string in self._joints_type[self._joints_dynamic]:
            if 'Left' in joint_id_string:
                joint_id_string_wo.append(joint_id_string[4:])
            elif 'Right' in joint_id_string:
                joint_id_string_wo.append(joint_id_string[5:])
            else:
                joint_id_string_wo.append(joint_id_string)
        unique_strings = list(dict.fromkeys(joint_id_string_wo))
        joint_ids = [unique_strings.index(s) for s in joint_id_string_wo]
        return torch.tensor(joint_ids)

    def to_position(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_kinematics(x)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        n = q.shape[0]
        j_n = self._offsets.shape[0]
        offsets = self._offsets.to(q.device)
        p3d = list()
        q_f = list()
        for i in np.arange(0, j_n):
            if self._parents[i] > -1:
                q_f.append(Quaternion.mul_(q_f[self._parents[i]], q[:, i]))
                p3d.append(Quaternion.rotate_(q_f[self._parents[i]], offsets[i] - offsets[self._parents[i]]) + p3d[self._parents[i]])
            else:
                q_0 = torch.zeros(n, 4, device=q.device)
                q_0[:, 0] = 1.
                q_f.append(q_0)
                p3d.append(torch.zeros(n, 3, device=q.device))
        return torch.stack(p3d, dim=1)
        with torch.no_grad():
            bm = BodyModel(self._body_model_path, batch_size=rotations.shape[0])
            return bm(
                pose_body=Quaternion(rotations[:, 1:].cpu()).axis_angle.flatten(start_dim=-2),
            ).Jtr[:, :22]

    def _compute_metadata(self):
        self._has_children = torch.zeros(len(self._parents), dtype=torch.bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._compute_chain_list()

    def _compute_chain_list(self):
        self._chain_list = []
        dyn_parents = self.dynamic_parents()
        for i, parent in enumerate(dyn_parents):
            if parent == -1:
                self._chain_list.append([i])
            else:
                self._chain_list.append(1 * [i])
                #if i < 22:#if (i >= 11 and i <= 13) or (i <= 2):
                #    self._chain_list.append(10 * [i])
                #else:
                #    self._chain_list.append([i])
                #self._chain_list.append(self._chain_list[parent] + [i])

    def dynamic_parents(self):
        new_parents = self._parents.clone()
        d = 0
        for i, parent in enumerate(self._parents):
            if not i in self.dynamic_nodes:
                new_parents[i] = -2
                new_parents[new_parents>=(i-d)] = new_parents[new_parents>=(i-d)] - 1
                d += 1

        return new_parents[new_parents != -2]












