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
    epoch_num = 20

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
        data = UserAction_Dataset.default_init()
        train_len = int(len(data) * sampling_percent)
        if rand_sampling:
            # 随机取样
            train, test = random_split(data, [train_len, len(data) - train_len])
        else:
            train, test = data[:train_len], data[train_len:]
            train = UserAction_Dataset(train[0], train[1])
            test = UserAction_Dataset(test[0], test[1])
            pass
        self.train_loader = DataLoader(train)
        self.test_loader = DataLoader(test)
        self.model.to(self.device)
        pass

    def train(self):
        from tqdm import tqdm
        batch = self.train_loader
        # 开始训练
        print("Training......")
        for e in range(self.epoch_num):
            with tqdm(total=len(batch), desc="epoch:{}".format(str(e)), position=0) as bar:
                for i, data in enumerate(batch):
                    # forward
                    inputs, labels = data
                    inputs = inputs.view(len(inputs), 1, -1)
                    outputs = self.model(inputs)

                    # Compute loss
                    self.optimizer.zero_grad()
                    Loss = self.loss(outputs, labels)

                    # backward
                    Loss.backward()

                    # update weights
                    self.optimizer.step()

                    bar.update(1)
            if e % 10 == 0:
                print('Epoch: {:4}, Loss: {:.5f}'.format(e, Loss.item()))
        pass

    def save(self, path="./model.pkl"):
        pickle.dump(self.model, open(path, 'wb'))

    @staticmethod
    def load(path="./model.pkl"):
        return pickle.load(open(path, 'rb'))


if __name__ == '__main__':
    obj = UserAction_run()
    obj.train()
    obj.save()

    pass
