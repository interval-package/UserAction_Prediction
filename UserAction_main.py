import logging
import argparse

from UserAction_Run import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--verbose", help="increase output verbosity",
    action="store_true"
)
parser.add_argument(
    "--logpath", action="store",
    default='./runs/logs/model.log',
    help="edit log saving path"
)
parser.add_argument(
    "--load", action="store",
    type=bool, default=False,
    help="load model from file"
)
parser.add_argument(
    "--modelpath", action="store",
    default="./data/model.pkl",
    help="edit model saving path"
)
parser.add_argument(
    "-e", "--epochs", action="store",
    type=int, default=20,
    help="train epoch"
)
parser.add_argument(
    "--cuda", action="store",
    type=bool, default=False,
    help="using cuda"
)
parser.add_argument(
    "--test", action="store_true",
    type=bool, default=True,
    help="enable to predict"
)
parser.add_argument(
    "--train", action="store_true",
    type=bool, default=True,
    help="enable to train"
)

if __name__ == '__main__':

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s:: %(message)s',
                        level=logging.INFO if args.verbose else logging.WARNING,
                        filename=args.logpath,
                        filemode='a')
    if args.load:
        obj = UserAction_run()
    else:
        obj = UserAction_run.loading_init(args.modelpath)

    if args.cuda:
        obj.device = torch.device("cuda")
        obj.model.device = obj.device

    if args.train:
        obj.train()
        obj.save(args.modelpath)

    if args.test:
        obj.test()
    pass
