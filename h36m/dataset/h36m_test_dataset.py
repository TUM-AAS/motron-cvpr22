from typing import List

import numpy as np
from h36m.dataset.h36m_torch_dataset import H36MTorchDataset

all_actions = ['walking', 'eating', 'smoking', 'discussion', 'directions', 'greeting', 'phoning', 'posing', 'purchases',
               'sitting', 'sittingdown', 'takingphoto', 'waiting', 'walkingdog', 'walkingtogether']


class H36MTestDataset(H36MTorchDataset):
    def __init__(self,
                 dataset,
                 subjects: List[str],
                 history_length: int,
                 prediction_horizon: int,
                 action: str,
                 num_samples: int = 8,
                 **kwargs):
        self.num_samples = num_samples
        self.actions = [action]
        if action == 'average':
            self.actions = all_actions
        super().__init__(dataset, subjects, history_length, prediction_horizon)

    def index_data(self):
        for action in self.actions:
            rnd = np.random.RandomState(1234567890)
            subject = self.subjects[0]
            for i in range(self.num_samples // 2):
                idx = rnd.randint(16, len(self._data[subject][action]['1_d0']['trajectory']) - 150) + 50 - self.history_length
                self.dataset_index += [(subject,
                                        action,
                                        '1_d0',
                                        idx
                                        )
                                       ]
                idx = rnd.randint(16, len(
                    self._data[subject][action]['2_d0']['trajectory']) - 150) + 50 - self.history_length
                self.dataset_index += [(subject,
                                        action,
                                        '2_d0',
                                        idx
                                        )
                                       ]

    def __len__(self):
        return self.num_samples * len(self.actions)
