from torch.utils.data import Dataset, DataLoader, random_split
from data.build_dataset import get_label
import numpy as np
import pandas as pd
import torch
import pickle


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
        if is_pkl:
            temp = pickle.load(open("./data/trace.pkl", "rb"))
        else:
            temp = pd.read_csv("./data/trace.csv")
        source = temp.values[:, :-1]
        label = temp.values[:, -1]
        return cls(source, label)

    @classmethod
    def by_day_init(cls):
        """
        使用正像数据进行划分

        :return: train dataset, test dataset
        """
        df_temp = pd.read_csv("./data/trace.csv")
        df_train = df_temp[(df_temp['day'] >= 1) & (df_temp['day'] <= 4)]
        df_test = df_temp[df_temp['day'] == 5]
        df_test = df_test[df_test['is_free'] == 1]
        return cls(df_train.values[:, :-1], df_train[:, -1]), cls(df_test.values[:, :-1], df_test[:, -1])

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

    temp = pd.read_csv("./data/trace.csv")
    pickle.dump(temp, open("./data/trace.pkl", "wb"))

    pass
