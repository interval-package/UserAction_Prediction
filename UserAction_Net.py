import torch
import torch.nn as nn


# user id using embedding 转化为低稠密类型


class UserAction_Net(nn.Module):
    def __init__(self, inp_dim=5,
                 out_dim=1,
                 mid_dim=8,
                 mid_layers=1):
        super(UserAction_Net, self).__init__()

        self.LSTM = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.LSTM(x)[0]  # y, (h, c) = self.rnn(x)
        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    pass

