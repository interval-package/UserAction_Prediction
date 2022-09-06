import pickle

from UserAction_Net import *
from UserAction_Dataset import UserAction_Dataset
from torch.utils.data import DataLoader, random_split
import pickle as pkl


class UserAction_run:
    __doc__ = """
        lr = 1e-2
        epoch_num = 1000
    """

    # define params
    lr = 1e-2
    epoch_num = 1000

    # define device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss func
    loss = nn.MSELoss()

    def __init__(self, model: nn.Module = None,
                 sampling_percent: float = 0.9,
                 rand_sampling: bool = False):
        """
        :cvar
        """
        if model is not None:
            self.model = model
        else:
            self.model = UserAction_Net()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # sampling set
        data = UserAction_Dataset()
        train_len = int(len(data) * sampling_percent)
        if rand_sampling:
            # 随机取样
            train, test = random_split(data, [train_len, len(data) - train_len])
        else:
            train, test = data[:train_len], data[train_len:]
            pass
        train_loader = DataLoader(train)
        test_loader = DataLoader(test)

        pass

    def train(self):
        batch = DataLoader(dataset=UserAction_Dataset())
        # 开始训练
        print("Training......")
        for e in range(self.epoch_num):
            for i, data in enumerate(batch):
                print(i, data)
                # forward
                inputs, labels = data
                outputs = self.model(inputs)

                # Compute loss
                with self.optimizer.zero_grad():
                    Loss = self.loss(outputs, labels)

                    # backward
                    Loss.backward()

                    # update weights
                    self.optimizer.step()

                if e % 10 == 0:
                    print('Epoch: {:4}, Loss: {:.5f}'.format(e, Loss.item()))
        pass

    def save(self, path="./model.pkl"):
        pickle.dump(self.model, open(path, 'wb'))

    def load(self):
        pass


if __name__ == '__main__':

    obj = UserAction_run()
    obj.train()
    obj.save()

    pass
