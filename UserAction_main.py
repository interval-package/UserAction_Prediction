import logging
import argparse

from UserAction_Run import *


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG,
                        filename='./runs/logs/model.log',
                        filemode='a')
    obj = UserAction_run()
    obj.train()
    obj.save()
    obj.test()
    pass
