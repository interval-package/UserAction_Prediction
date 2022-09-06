from paddle.fluid.dataloader import random_split
from torch.utils.data import Dataset, DataLoader
from data.build_dataset import get_label
import numpy as np
import pandas as pd
import torch


class UserAction_Dataset(Dataset):
    __doc__ = r"""
    the UserAction_Dataset
    at the fmt of
    dtype = np.dtype([("id", "i4"), ("time", "f4"), ("screen_lock_status", "i1"), ("network_status", "i1"),
                      ("battery_charge_status", "i1"), ("screen_status", "i1"), ("battery_level", "i1"),
                      ("idle", "i1")])
    thus we aiming at using the historical data the predict next time action                 
    
    """

    dtype = np.dtype([("id", "i4"), ("time", "f4"), ("screen_lock_status", "i1"), ("network_status", "i1"),
                      ("battery_charge_status", "i1"), ("screen_status", "i1"), ("battery_level", "i1"),
                      ("idle", "i1")])

    def __init__(self):
        super(UserAction_Dataset, self).__init__()
        self.source = np.load("./data/temp/trace.npy",
                              mmap_mode=None,
                              allow_pickle=True,
                              fix_imports=True)

        self.label = np.load("./data/temp/label_arr.npy",
                             mmap_mode=None,
                             allow_pickle=True,
                             fix_imports=True).astype(int)
        pass

    def __getitem__(self, index):
        return self.source[index], self.label[index]

    def __len__(self):
        return len(self.source)


# class UserAction_Dataloader(DataLoader):
#     pass


if __name__ == '__main__':
    # from torch.utils.data import random_split
    #
    tar = UserAction_Dataset()
    print(tar[0])

    train_percent = int(len(tar)*0.7)
    train, test = random_split(tar, [train_percent, len(tar) - train_percent])
    train_loader = DataLoader(train)
    test_loader = DataLoader(test)

    for arr, y in train_loader:
        print(arr, y)

    # print(temp[0])
    # print(get_label(obj.source))
    pass
