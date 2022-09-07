"""

"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from UserAction_Net import *
from UserAction_Dataset import UserAction_Dataset
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import pickle


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
                 sampling_type: str = 'part'):
        """
        sampling_type: str {'part', 'rand'}
        'part' would straightly split the dataset in the midlle
        'rand' would call rand sampling methods
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
        if sampling_type is 'rand':
            # 随机取样
            train, test = random_split(data, [train_len, len(data) - train_len])
        elif sampling_type is 'part':
            train, test = data[:train_len], data[train_len:]
            train = UserAction_Dataset(train[0], train[1])
            test = UserAction_Dataset(test[0], test[1])
            pass
        else:
            raise ValueError("unrecognized type for sampling")
        self.train_loader = DataLoader(train)
        self.test_loader = DataLoader(test)
        self.model.to(self.device)
        pass

    @classmethod
    def loading_init(cls):
        """
        loading trained model from file
        :return:
        """
        return cls(cls.load(), sampling_type='part')

    def train(self):
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

    def save(self, path="./data/model.pkl"):
        pickle.dump(self.model, open(path, 'wb'))

    @staticmethod
    def load(path="./data/model.pkl"):
        return pickle.load(open(path, 'rb'))

    # 模型测试
    def test(self):
        print('test model')
        model = self.model.to(self.device)
        model.eval()
        avg_acc = 0
        with tqdm(total=len(self.test_loader), desc="test process:", position=0) as bar:
            for i, data in enumerate(self.test_loader):
                x, y_test = data
                pred = model(x)
                df = self.get_accuracy(y_test, pred)
                print(df)
                # avg_acc += df["验证集准确率"]
                bar.update(1)
                pass
        # avg_acc = avg_acc / len(self.test_loader)
        return avg_acc

    @staticmethod
    def get_accuracy(y_test, y_x):
        df = pd.DataFrame()
        df["验证集准确率"] = "{:.2f}%".format(accuracy_score(y_test, y_x) * 100)
        df["验证集精确率"] = "{:.2f}%".format(precision_score(y_test, y_x, average='macro') * 100)  # 打印验证集查准率
        df["验证集召回率"] = "{:.2f}%".format(recall_score(y_test, y_x, average='macro') * 100)  # 打印验证集查全率
        df["验证集F1值"] = "{:.2f}%".format(f1_score(y_test, y_x, average='macro') * 100)
        return df


if __name__ == '__main__':
    obj = UserAction_run()
    obj.train()
    obj.save()
    pass
