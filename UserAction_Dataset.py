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
        self.source = torch.tensor(source, dtype=torch.float)
        self.label = torch.tensor(label,dtype=torch.float)
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

    def split(self, percent: float):
        return


# class UserAction_Dataloader(DataLoader):
#     pass


if __name__ == '__main__':
    # from torch.utils.data import random_split

    sampling_percent = 0.7
    rand_sampling = False

    data = UserAction_Dataset.default_init()
    train_len = int(len(data) * sampling_percent)
    if rand_sampling:
        # 随机取样
        train, test = random_split(data, [train_len, len(data) - train_len])
    else:
        train, test = data[:train_len], data[train_len:]
        train = UserAction_Dataset(train[0], train[1])
        test = UserAction_Dataset(test[0], test[1])
        pass
    train_loader = DataLoader(train)
    test_loader = DataLoader(test)

    for i in train_loader:
        print(i)

    pass
