import logging
import argparse

from UserAction_Run import *


parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("--logpath", action="store",
                    default='./runs/logs/model.log',
                    help="edit log saving path")
parser.add_argument("--modelpath",action="store",
                    default="./data/model.pkl",
                    help="edit model saving path")
parser.add_argument("-e", "--epochs", action="store",
                    type=int, default=20,
                    help="train epoch")

if __name__ == '__main__':

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s:: %(message)s',
                        level=logging.INFO if args.verbose else logging.WARNING,
                        filename=args.logpath,
                        filemode='a')
    obj = UserAction_run()
    obj.train()
    obj.save(args.modelpath)
    obj.test()
    pass
