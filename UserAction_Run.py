"""

"""
import logging

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from UserAction_Net import *
from UserAction_Dataset import UserAction_Dataset, split_nag_pos
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
                 sampling_percent: float = None,
                 sampling_type: str = 'part'):
        """
        :cvar
        sampling_type: str {'part', 'rand'}: for specific type of sampling
            'part': would straightly split the dataset in the middle
            'rand': would call rand sampling methods
        """

        self.test_loader = None
        self.train_loader = None
        self.sampling_percent = sampling_percent
        self.sampling_type = sampling_type

        if model is not None:
            self.model = model
        else:
            self.model = UserAction_Net()

        self.model.device = self.device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.to(self.device)

        logging.info("pre-building finished, with {} split method at {}.".format(
            sampling_type,
            "not" if sampling_percent is None else sampling_percent))
        logging.info("run device: {}, net device: {}".format(self.device, self.model.device))

    @classmethod
    def loading_init(cls, path, sampling_type):
        """
        loading trained model from file
        :return: model load from file
        """
        return cls(cls.load(path), sampling_type=sampling_type)

    def train_(self):
        """
        更改了训练的方式，老方法是一条一条数据放进去训练，现在是将数据按照一个一个seq放放进去，同时一次放置多个数据
        也就是我们现在每次放入的tensor为[N,L,H]，N为我们的样本数量，L为我们设定的一个时间序列的长度，H是在序列中每个单维样本的长度

        这里我们将第一个列id分割出来，进行一个embedding处理
        :return:
        """
        # 开始训练
        # print('start to training model......')
        logging.info('start to training model......')
        self.train_loader.dataset.reshape()
        for e in range(self.epoch_num):
            with tqdm(self.train_loader, desc="epoch:{}".format(str(e)), position=0) as t_epoch:
                for inputs, label in t_epoch:
                    outputs = self.model(inputs)

                    # Compute loss
                    self.optimizer.zero_grad()
                    Loss = self.loss(outputs, label)

                    # backward
                    Loss.backward()

                    self.optimizer.step()

                    t_epoch.set_postfix(Loss=Loss.item())
                    pass
        pass

    def train_old(self):
        # 开始训练
        # print('start to training model......')
        logging.info('start to training model......')
        for e in range(self.epoch_num):
            with tqdm(self.train_loader, desc="epoch:{}".format(str(e)), position=0) as t_epoch:
                for inputs, labels in t_epoch:
                    self.model.zero_grad()

                    # forward
                    inputs = inputs.squeeze()
                    outputs = self.model(inputs)

                    # Compute loss
                    self.optimizer.zero_grad()
                    # labels = labels.to(self.device)
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
        logging.info("saving model to " + path)
        pickle.dump(self.model, open(path, 'wb'))

    @staticmethod
    def load(path="./data/model.pkl"):
        logging.info("loading model from " + path)
        return pickle.load(open(path, 'rb'))

    # 模型测试
    def test(self):
        logging.info('start to testing model......')
        self.model.eval()
        y_x, y_test = [], []
        with tqdm(self.test_loader, desc="pred and test process:", position=0) as bar:
            with torch.no_grad():
                for x, y in bar:
                    pred = self.bulk_predict(x)
                    y_x.append(pred.item())
                    y_test.append(y.item())
                    pass
                res = self.get_accuracy(y_test, y_x)
        logging.info("finished testing!")
        logging.info(res.__str__())
        return res

    def test_by_nag_pos(self):
        # if self.test_loader is None:
        #     raise ValueError("test loader not inited.")

        data = split_nag_pos()

        for key in ["df_test_n0", "df_test_n1"]:
            logging.info("start {}".format(key))
            with tqdm(torch.as_tensor(data[key].values, dtype=torch.float),
                      desc="{} process:".format(key),
                      position=0) as bar:

                y_x, y_test = [], []
                with torch.no_grad():
                    for x in bar:
                        # x = x.values
                        pred = self.model.predict(x[:-1].view(1, 1, -1)).int()
                        y_x.append(pred.item())
                        y_test.append(x[-1])
                        pass
                    res = self.get_accuracy(y_test, y_x)
                pass

            logging.info("finished testing! for {}".format(key))
            logging.info(key, res.__str__())
            pass

        pass

    def bulk_predict(self, inputs):
        """
        :param inputs: 也就是输入的参数，在这里会被展开，才能传入网络
        :return: 这里网络输出并不是最直接预测的结果，我们使用概率最大作为可能
        """
        inputs = inputs.view(len(inputs), 1, -1)
        # inputs = torch.unsqueeze(inputs, 0)
        # outputs = self.model(inputs).max(dim=1)[1]
        outputs = self.model.predict(inputs).int()
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

    def generate_dataset(self):
        # sampling set
        if self.sampling_type == 'by_day':
            train, test = UserAction_Dataset.by_day_init()

        elif self.sampling_type == 'rand':
            # 随机取样
            data = UserAction_Dataset.default_init()
            train, test = data.split_rand_sample(self.sampling_percent)
            del data

        elif self.sampling_type == 'part':
            # 直接采样，直接将数据集分割为两部分
            data = UserAction_Dataset.default_init()
            train, test = data.split_part_sample(self.sampling_percent)
            del data
        else:
            raise ValueError("unrecognized type for sampling")

        train.change_device(self.device)
        test.change_device(self.device)

        self.train_loader = DataLoader(train)
        self.test_loader = DataLoader(test)
        return


if __name__ == '__main__':
    obj = UserAction_run()
    # obj.train()
    # obj.save()
    # obj.test()
    pass
