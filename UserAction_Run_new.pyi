import torch
from torch import nn

from UserAction_Dataset import UserAction_Dataset, split_nag_pos
from UserAction_Net import UserAction_Net


class UserAction_Run_new:


    # define params
    lr = 1e-2
    epoch_num = 20

    # define device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss func
    loss = nn.MSELoss()


    def __init__(self, model):
        self.data = split_nag_pos()

        if model is not None:
            self.model = model
        else:
            self.model = UserAction_Net()

        self.model.device = self.device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

        pass


    def train(self, action="n"):
        pass

    def test(self, action="n"):
        pass

    pass