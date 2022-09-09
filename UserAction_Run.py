"""

"""
import logging

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from UserAction_Net import *
from UserAction_Dataset import UserAction_Dataset
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import pickle


class UserAction_run:
    __doc__ = """
        there are some build in params:
            lr = 1e-2; the learning rate
            epoch_num = 20; training epoch times
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
        :cvar
        sampling_type: str {'part', 'rand'}: for specific type of sampling
            'part': would straightly split the dataset in the middle
            'rand': would call rand sampling methods
        """

        if model is not None:
            self.model = model
        else:
            self.model = UserAction_Net()

        UserAction_Net.device = self.device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # sampling set
        data = UserAction_Dataset.default_init()
        if sampling_type == 'rand':
            # 随机取样
            train, test = data.split_rand_sample(sampling_percent)
        elif sampling_type == 'part':
            # 直接采样，直接将数据集分割为两部分
            train, test = data.split_part_sample(sampling_percent)
            pass
        else:
            raise ValueError("unrecognized type for sampling")
        del data

        self.train_loader = DataLoader(train)
        self.test_loader = DataLoader(test)
        self.model.to(self.device)

        logging.info("pre-building finished, with {} split method at {:2f}.".format(sampling_type, sampling_percent))
        logging.info("using device: {}".format(self.device))
        pass

    @classmethod
    def loading_init(cls, path):
        """
        loading trained model from file
        :return: model load from file
        """
        return cls(cls.load(path), sampling_type='part')

    def train(self):
        batch = self.train_loader
        # 开始训练
        print('start to training model......')
        logging.info('start to training model......')
        for e in range(self.epoch_num):
            with tqdm(batch, desc="epoch:{}".format(str(e)), position=0) as t_epoch:
                for inputs, labels in t_epoch:
                    # forward
                    # inputs, labels = data
                    inputs = inputs.view(len(inputs), 1, -1)
                    inputs.to(self.device)
                    outputs = self.model(inputs)

                    # Compute loss
                    self.optimizer.zero_grad()
                    Loss = self.loss(outputs, labels)

                    # backward
                    Loss.backward()

                    # update weights
                    self.optimizer.step()

                    t_epoch.set_postfix(Loss=Loss.item())
            if e % 10 == 0:
                logging.info('Epoch: {:4}, Loss: {:.5f}'.format(e, Loss.item()))
        pass

    def save(self, path="./data/model.pkl"):
        pickle.dump(self.model, open(path, 'wb'))

    @staticmethod
    def load(path="./data/model.pkl"):
        return pickle.load(open(path, 'rb'))

    # 模型测试
    def test(self):
        logging.info('start to testing model......')
        self.model.eval()
        y_x, y_test = [], []
        with tqdm(self.test_loader, desc="pred and test process:", position=0) as bar:
            with torch.no_grad():
                for x, y in bar:
                    pred = self.predict(x)
                    y_x.append(pred.item())
                    y_test.append(y.item())
                    pass
                res = self.get_accuracy(y_test, y_x)
        logging.info("finished testing!")
        logging.info(res.__str__())
        return res

    def predict(self, inputs):
        """
        :param inputs: 也就是输入的参数，在这里会被展开，才能传入网络
        :return: 这里网络输出并不是最直接预测的结果，我们使用概率最大作为可能
        """
        inputs = inputs.view(len(inputs), 1, -1)
        # inputs = torch.unsqueeze(inputs, 0)
        # outputs = self.model(inputs).max(dim=1)[1]
        outputs = self.model(inputs).int()
        return outputs

    @staticmethod
    def get_accuracy(y_test, y_x) -> dict:
        logging.info("getting accuracy")
        res = dict()
        res["accuracy_score"] = accuracy_score(y_test, y_x) * 100
        res["precision_score"] = precision_score(y_test, y_x, average='macro') * 100  # 打印验证集查准率
        res["recall_score"] = recall_score(y_test, y_x, average='macro') * 100  # 打印验证集查全率
        res["f1_score"] = f1_score(y_test, y_x, average='macro') * 100
        return res


if __name__ == '__main__':
    # obj = UserAction_run()
    # obj.train()
    # obj.save()
    # obj.test()
    pass
