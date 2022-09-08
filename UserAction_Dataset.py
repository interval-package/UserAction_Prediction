from torch.utils.data import Dataset, DataLoader, random_split
from data.build_dataset import get_label
import numpy as np
import pandas as pd
import torch


class UserAction_Dataset(Dataset):
    __doc__ = r"""
    the UserAction_Dataset
    at the fmt of
    thus we aiming at using the historical data the predict next time action
    """

    def __init__(self, source, label):
        super(UserAction_Dataset, self).__init__()
        self.source = torch.as_tensor(source, dtype=torch.float)
        self.label = torch.as_tensor(label, dtype=torch.float)
        pass

    @classmethod
    def default_init(cls):
        """
        default loading all the data without sampling
        :return:
        """
        source = pd.read_csv("./data/trace.csv").values[:, :-1]

        label = np.load("./data/temp/label_arr.npy",
                        mmap_mode=None,
                        allow_pickle=True,
                        fix_imports=True)
        return cls(source, label)

    def __getitem__(self, index):
        return self.source[index], self.label[index]

    def __len__(self):
        return len(self.source)

    def split_rand_sample(self, sampling_percent: float):
        train_len = int(len(data) * sampling_percent)
        train, test = random_split(self, [train_len, len(self) - train_len])
        return train, test

    def split_part_sample(self, sampling_percent: float):
        train_len = int(len(self) * sampling_percent)
        train, test = self[:train_len], self[train_len:]
        train = UserAction_Dataset(train[0], train[1])
        test = UserAction_Dataset(test[0], test[1])
        return train, test

    def split_order_sample(self, sampling_percent: float):
        return


if __name__ == '__main__':

    data = UserAction_Dataset.default_init()

    pass
