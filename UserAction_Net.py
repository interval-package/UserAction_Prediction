import torch
import torch.nn as nn
import UserAction_Dataset


# user id using embedding 转化为低稠密类型


class UserAction_Net(nn.Module):
    # define device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, embedding_size=6, inp_len=12, out_dim=1, mid_dim=1, mid_layers=2):
        super(UserAction_Net, self).__init__()
        self.embedding = nn.Embedding(UserAction_Dataset.batch_size, embedding_size)
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
        idx, features = x.split(1,self.feature_len)
        idx = self.embedding(idx).squeeze()
        x = torch.cat(idx, features)
        y = self.LSTM(x)[0]  # y, (h, c) = self.rnn(x)
        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        y = y.reshape(1)
        return y

    pass
