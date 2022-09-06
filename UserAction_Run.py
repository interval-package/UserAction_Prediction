from UserAction_Net import *
from UserAction_Dataset import UserAction_Dataset
from torch.utils.data import DataLoader, random_split
import pickle as pkl


class UserAction_run:
    __doc__ = """
    
    """

    lr = 1e-2
    # define device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss func
    loss = nn.MSELoss()

    def __init__(self, model: nn.Module):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        tar = UserAction_Dataset()
        train_percent = int(len(tar) * 0.7)
        train, test = random_split(tar, [train_percent, len(tar) - train_percent])
        train_loader = DataLoader(train)
        test_loader = DataLoader(test)

        pass

    epoch_num = 1000

    def train(self, _dataloader):
        batch = DataLoader(dataset=UserAction_Dataset())
        # 开始训练
        print("Training......")
        for e in range(self.epoch_num):
            out = self.model(_dataloader)

            Loss = self.loss(out, batch)

            self.optimizer.zero_grad()
            Loss.backward()
            self.optimizer.step()

            if e % 10 == 0:
                print('Epoch: {:4}, Loss: {:.5f}'.format(e, Loss.item()))
        pass


if __name__ == '__main__':
    pass
