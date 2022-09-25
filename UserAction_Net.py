import torch
import torch.nn as nn
from UserAction_Dataset import user_size, seq_len


# user id using embedding 转化为低稠密类型


class UserAction_Net(nn.Module):
    # define device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
        记录时间信息进去
    """

    h = None

    def __init__(self, embedding_size=6, inp_len=11, out_dim=1, mid_dim=10, mid_layers=2):
        super(UserAction_Net, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(user_size, embedding_size)
        # embedding the id

        self.inp_dim = inp_len + embedding_size - 1
        self.feature_len = inp_len - 1

        self.LSTM = nn.LSTM(self.inp_dim, mid_dim, mid_layers)  # rnn

        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        idx, features = x.split([1, self.feature_len], dim=2)
        idx = self.embedding(idx.int()).view(-1, seq_len, self.embedding_size)
        x = torch.cat([idx, features], dim=2)

        y, (h, c) = self.LSTM(x, self.h)  # y, (h, c) = self.rnn(x)

        # h, c = h

        # LSTM中，训练的时候，需要把上次一的隐藏状态传入。
        #
        # 但是传入的数据需要重新计算backward,去更新梯度。
        #
        # 这就造成对一个变量进行了多次backward。

        self.h = (h.data, c.data)
        y = self.reg(y)
        return y

    def predict(self, x):
        idx, features = x.split([1, self.feature_len], dim=2)
        idx = self.embedding(idx.int()).view(-1, 1, self.embedding_size)
        x = torch.cat([idx, features], dim=2)
        y, (h, c) = self.LSTM(x, None)  # y, (h, c) = self.rnn(x)
        y = self.reg(y)
        return y

    pass
