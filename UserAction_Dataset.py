from torch.utils.data import Dataset, DataLoader, random_split
from data.build_dataset import get_label
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle

_method = "sql"
# range(1000) in [0,999]
batch_size = 1000
seq_len = 11


def load_df(method=_method):
    if method == "sql":
        import sqlite3
        con = sqlite3.connect("./data/UserAction.db")
        df = pd.read_sql_query(sql="select * from trace", con=con)
    elif method == "csv":
        df = pd.read_csv("./data/trace.csv")
    else:
        raise ValueError("method not fit")
    # df.drop(["idle"])
    return df


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

    def change_device(self, dev):
        """
        we should transport the dataset to the device as we init the dataset

        :param dev: target device
        :return:
        """
        self.source = self.source.to(dev)
        self.label = self.label.to(dev)
        pass

    @classmethod
    def default_init(cls, is_pkl=False):
        """
        default loading all the data without sampling
        :return:
        """
        temp = load_df()
        source = temp.values[:, :-1]
        label = temp.values[:, -1]
        return cls(source, label)

    @classmethod
    def by_day_init(cls):
        """
        使用正像数据进行划分

        :return: train dataset, test dataset
        """
        df_temp = load_df()
        df_train = df_temp[(df_temp['day'] >= 1) & (df_temp['day'] <= 4)]
        df_test = df_temp[df_temp['day'] == 5]
        df_test = df_test[df_test['is_free'] == 1]
        return cls(df_train.values[:, :-1], df_train.values[:, -1]), cls(df_test.values[:, :-1], df_test.values[:, -1])

    def __getitem__(self, index):
        return self.source[index], self.label[index]

    def __len__(self):
        return len(self.source)

    def split_rand_sample(self, sampling_percent: float):
        train_len = int(len(self) * sampling_percent)
        train, test = random_split(self, [train_len, len(self) - train_len])
        return train.dataset, test.dataset

    def split_part_sample(self, sampling_percent: float):
        train_len = int(len(self) * sampling_percent)
        train, test = self[:train_len], self[train_len:]
        train = UserAction_Dataset(train[0], train[1])
        test = UserAction_Dataset(test[0], test[1])
        return train, test

    def split_order_sample(self, sampling_percent: float):
        return


if __name__ == '__main__':
    ts = torch.tensor([10, 1, 2, 3])
    ts_1, ts_2 = ts.split([1, 3])
    embed = nn.Embedding(batch_size, 3, 1)
    vec = embed(ts_1).squeeze()
    print(torch.cat([vec, ts_2]))
    pass
