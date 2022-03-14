import torch
from torch.utils import data


class AMASSTorchDataset(data.Dataset):
    def __init__(self,
                 index,
                 pose_data,
                 history_length: int,
                 prediction_horizon: int,
                 trans_data=None,
                 transform=lambda x: x,
                 window=None,
                 **kwargs):
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.transform = transform

        self._index = index
        self._pose_data = pose_data
        self._trans_data = trans_data
        self._window = window


        self.dataset_index = []

        self.index_data()

    def index_data(self):
        seq_lengths = self._index[:, 1] - self._index[:, 0]
        for j in range(seq_lengths.shape[0]):
            if self._window:
                for i in range(0, seq_lengths[j] - self._window[0] + 1, self._window[1]):
                    self.dataset_index += [(j, i)]
            else:
                for i in range(seq_lengths[j] - self.history_length - self.prediction_horizon):
                    self.dataset_index += [(j, i)]

    def __getitem__(self, item):
        i_idx, i = self.dataset_index[item]
        ts_pose = self._pose_data[self._index[i_idx, 0] + i: self._index[i_idx, 0] + i + self.history_length + self.prediction_horizon]
        ts_pose = self.transform(ts_pose)
        ts_pose = torch.tensor(ts_pose)

        if hasattr(self, 'dataset_name'):
            dp = (self.dataset_name, i_idx, i)
        else:
            dp = ("_", i_idx, i)

        if self._trans_data is not None:
            ts_trans = self._trans_data[self._index[i_idx, 0] + i: self._index[i_idx, 0] + i + self.history_length + self.prediction_horizon]
            ts_trans = torch.tensor(ts_trans)
            return (ts_pose[:self.history_length],
                    ts_pose[self.history_length: self.history_length + self.prediction_horizon],
                    ts_trans[:self.history_length],
                    ts_trans[self.history_length: self.history_length + self.prediction_horizon])

        return (ts_pose[:self.history_length],
                ts_pose[self.history_length: self.history_length + self.prediction_horizon])

    def __len__(self):
        return len(self.dataset_index)

    def hparam(self) -> dict:
        return {
            'DATA_history_length': self.history_length,
            'DATA_prediction_horizon': self.prediction_horizon,
        }