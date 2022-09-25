import logging

from torch.utils.data import Dataset, random_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

"""
    部分预设的方式
"""

_method = "sql"

# range(1000) in [0,999]

from UserAction_main import user_size, feature_len, seq_len, window_size


def load_df(method=_method):
    logging.info("loading data by {}".format(_method))
    if method == "sql":
        import sqlite3
        con = sqlite3.connect("./data/UserAction.db")
        _df = pd.read_sql_query(sql="select * from trace", con=con)
    elif method == "csv":
        _df = pd.read_csv("./data/trace.csv")
    else:
        raise ValueError("method not fit")
    # df.drop(["idle"])
    return _df


class UserAction_Dataset(Dataset):
    __doc__ = r"""
    the UserAction_Dataset
    at the fmt of
    thus we aiming at using the historical data the predict next time action
    """

    # 基于数据集对对模型进行优化，考虑嵌入的过程
    # 有些改用整值的就应该拿去做embedding
    # 考虑一下，哪些feature需要进行一下embedding升维度的

    def __init__(self, source, label):
        super(UserAction_Dataset, self).__init__()
        self.source = torch.as_tensor(source, dtype=torch.float)
        self.label = torch.as_tensor(label, dtype=torch.float)
        pass

    def __getitem__(self, index):
        return self.source[index], self.label[index]

    def __len__(self):
        return len(self.source)

    def reshape(self):
        self.source = self.source.view(-1, seq_len, 11)
        self.label = self.label.view(-1, seq_len, 1)
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
    def default_init(cls):
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

    """
        之前使用每条数据作为样本进行训练，对于LSTM来说，我们输入的数据如果为序列数据的话会更好
        同时用大矩阵运算速度会加快
        
        进行时间序列化，为了使得我们能够准确的按照序列的长度划分，我们有两种处理方式：
        1.将训练集化为序列长的整数倍
        2.使用滑动窗口构建数据集
        
        目前而言是使用滑动窗口
    """

    train_len = 770000

    @classmethod
    def rand_sample_init(cls):
        """
        解决使用torch自带抽样为伪抽样，不划分数据的问题，使用numpy抽样，在构建对象时，就进行抽样，按照序列来进行划分

        :return:
        """
        return

    def split_rand_sample(self, sampling_percent: float = None):
        if sampling_percent is not None:
            train_len = int(len(self) * sampling_percent)
        else:
            train_len = self.train_len

        train, test = random_split(self, [train_len, len(self) - train_len])
        return train.dataset, test.dataset

    def split_part_sample(self, sampling_percent: float = None):
        if sampling_percent is not None:
            train_len = int(len(self) * sampling_percent)
        else:
            train_len = self.train_len

        train, test = self[:train_len], self[train_len:]
        train = UserAction_Dataset(train[0], train[1])
        test = UserAction_Dataset(test[0], test[1])
        return train, test


def split_nag_pos():
    df_temp = load_df()
    res = {}
    res["df_temp_5"] = df_temp[df_temp['day'] == 5]
    res["df_test_n0"] = df_temp[(df_temp['idle'] == 0) & (df_temp['day'] == 5)]
    res["df_test_n1"] = df_temp[(df_temp['idle'] == 1) & (df_temp['day'] == 5)]
    res["df_train_n0"] = df_temp[(df_temp['day'] >= 1) & (df_temp['day'] <= 4) & (df_temp['idle'] == 0)]
    res["df_train_n1"] = df_temp[(df_temp['day'] >= 1) & (df_temp['day'] <= 4) & (df_temp['idle'] == 1)]
    # TsN = len(df_temp_5)  # 验证集数量
    # N0 = len(df_test_n0)  # 负样本总数
    # N1 = len(df_test_n1)  # 正样本总数
    return res


if __name__ == '__main__':
    # torch 的dim属性是从高维到低维的
    # obj = UserAction_Dataset.default_init()
    # res = obj.source[:100]
    # res = obj.source[:10000]
    # print(res.view(-1, 10, 11))

    df = load_df()
    # df.set_index('id', inplace=True)
    df = df[df['id'] == 0]
    print(df)
    pass
