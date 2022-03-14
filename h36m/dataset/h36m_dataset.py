from collections import UserDict

import numpy as np
import torch


class H36MDataset(UserDict):
    def __init__(self, dataset_path: str, dataset_fps: int, dataset_downsample_factor: int = 1, **kwargs):
        self._fps = dataset_fps
        super().__init__(self._load(dataset_path))
        if dataset_downsample_factor > 1:
            self.downsample(dataset_downsample_factor)

    @staticmethod
    def _load(path):
        result = {}
        data_np = np.load(path, 'r', allow_pickle=True)
        for i, (trajectory, rotations, subject, action) in enumerate(zip(data_np['trajectories'],
                                                                         data_np['rotations'],
                                                                         data_np['subjects'],
                                                                         data_np['actions'])):
            if subject not in result:
                result[subject] = {}

            split_action_name = action.split('_')
            action_str = split_action_name[0]
            action_count = split_action_name[1]

            if action_str not in result[subject]:
                result[subject][action_str] = {}

            result[subject][action_str][action_count] = {
                'rotations': torch.tensor(rotations, dtype=torch.float),
                'trajectory': torch.tensor(trajectory, dtype=torch.float)
            }
        return result

    def downsample(self, factor, keep_strides=True):
        """
        Downsample this dataset by an integer factor, keeping all strides of the data
        if keep_strides is True.
        The frame rate must be divisible by the given factor.
        The sequences will be replaced by their downsampled versions, whose actions
        will have '_d0', ... '_dn' appended to their names.
        """
        assert self._fps % factor == 0

        for subject in self.data.keys():
            for action in list(self.data[subject].keys()):
                new_actions_idx = {}
                for action_idx in self.data[subject][action].keys():
                    for idx in range(factor):
                        tup = {}
                        for k in self.data[subject][action][action_idx].keys():
                            tup[k] = self.data[subject][action][action_idx][k][idx::factor]
                        new_actions_idx[action_idx + '_d' + str(idx)] = tup
                        if not keep_strides:
                            break
                self.data[subject][action] = new_actions_idx

        self._fps //= factor

    def __getitem__(self, item):
        return self.data[item]
