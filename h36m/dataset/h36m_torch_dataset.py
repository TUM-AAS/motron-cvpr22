from typing import List

import torch
from torch.utils import data

from h36m.dataset.h36m_dataset import H36MDataset
from motion import Skeleton, Quaternion


class H36MTorchDataset(data.Dataset):
    def __init__(self,
                 dataset: H36MDataset,
                 subjects: List[str],
                 history_length: int,
                 prediction_horizon: int,
                 transform=lambda x: x,
                 step: int = 1,
                 index_data: bool = True,
                 skip_11_d: bool = False,
                 **kwargs):
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.subjects = subjects
        self.transform = transform
        self.step = step
        self.skip_11_d = skip_11_d

        self._data = dataset
        self.dataset_index = []

        self._is_indexed = False

    def index_data(self):
        for subject in self.subjects:
            for action in self._data[subject].keys():
                for action_it in self._data[subject][action].keys():
                    if self.skip_11_d and subject == 'S11' and action == 'directions' and action_it == '2':
                        continue
                    for i in range(0, len(self._data[subject][action][action_it]['trajectory']) - self.history_length - self.prediction_horizon, self.step):
                        self.dataset_index += [(subject, action, action_it, i)]

    def __getitem__(self, item):
        # Check if data set is indexed otherwise index it
        if not self._is_indexed:
            self.index_data()
            self._is_indexed = True
        subject, action, action_idx, i = self.dataset_index[item]
        ts = self._data[subject][action][action_idx]['rotations'][i:i + self.history_length + self.prediction_horizon]
        ts = self.transform(ts)
        return ts[:self.history_length], ts[self.history_length: self.history_length + self.prediction_horizon]

    def __len__(self):
        # Check if data set is indexed otherwise index it
        if not self._is_indexed:
            self.index_data()
            self._is_indexed = True
        return len(self.dataset_index)

    def _mirror_sequence(self, sequence: torch.Tensor, skeleton: Skeleton):
        mirrored_rotations = sequence['rotations'].numpy().copy()
        mirrored_trajectory = sequence['trajectory'].numpy().copy()

        joints_left = skeleton._joints_left
        joints_right = skeleton._joints_right

        # Flip left/right joints
        mirrored_rotations[:, joints_left] = sequence['rotations'][:, joints_right]
        mirrored_rotations[:, joints_right] = sequence['rotations'][:, joints_left]

        mirrored_rotations[:, :, [2, 3]] *= -1
        mirrored_trajectory[:, 0] *= -1

        return {
            'rotations': Quaternion.qfix_(torch.tensor(mirrored_rotations, dtype=torch.float)),
            'trajectory': torch.tensor(mirrored_trajectory, dtype=torch.float)
        }

    def mirror(self, skeleton: Skeleton):
        """
        Perform data augmentation by mirroring every sequence in the dataset.
        The mirrored sequences will have '_m' appended to the action name.
        """
        for subject in self._data.keys():
            for action in list(self._data[subject].keys()):
                for action_idx in list(self._data[subject][action].keys()):
                    if '_m' in action:
                        continue
                    self._data[subject][action][action_idx + '_m'] = self._mirror_sequence(self._data[subject][action][action_idx], skeleton)

    def reverse(self):
        for subject in self._data.keys():
            for action in list(self._data[subject].keys()):
                for action_idx in list(self._data[subject][action].keys()):
                    if '_r' in action:
                        continue
                    self._data[subject][action][action_idx + '_r'] = self._reverse_sequence(self._data[subject][action][action_idx])

    def _reverse_sequence(self, sequence: torch.Tensor):
        # Reverse joints
        reversed_rotations = sequence['rotations'].clone().flip(0)
        reversed_trajectory = sequence['rotations'].clone().flip(0)

        return {
            'rotations': reversed_rotations,
            'trajectory': reversed_trajectory
        }

    def hparam(self) -> dict:
        return {
            'DATA_history_length': self.history_length,
            'DATA_prediction_horizon': self.prediction_horizon,
        }