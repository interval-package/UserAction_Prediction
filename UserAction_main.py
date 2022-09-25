import logging
import argparse

import UserAction_Dataset
from UserAction_Run import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--verbose", help="increase output verbosity",
    action="store_true"
)
parser.add_argument(
    "--savelog", action="store_true",
)
parser.add_argument(
    "--logpath", action="store",
    default='./runs/logs/model.log',
    help="edit log saving path"
)
parser.add_argument(
    "--load", action="store_true",
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
    "--gpu", action="store_true",
    help="using gpu, without this will auto detect"
)
parser.add_argument(
    "--test", action="store_true",
    help="enable to predict"
)
parser.add_argument(
    "--train", action="store_true",
    help="enable to train"
)

parser.add_argument(
    "--sampling", action="store",
    type=str, default=None,
    help="split method for the Dataset"
)

parser.add_argument(
    "--seqlen", action="store",
    type=int, default=1,
    help="sequence length for input size"
)

if __name__ == '__main__':

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s:: %(message)s',
                        level=logging.INFO if args.verbose else logging.WARNING,
                        filename=args.logpath if args.savelog else None,
                        filemode='a')
    logging.info("Program start....., detail in {}".format(args.logpath))
    if args.load:
        logging.info("loading models from {}".format(args.modelpath))
        obj = UserAction_run.loading_init(args.modelpath, args.sampling)
    else:
        logging.info("new model build, sampling {}".format(args.sampling))
        obj = UserAction_run(sampling_type=args.sampling)

    UserAction_Dataset.seq_len = args.seqlen

    obj.epoch_num = args.epochs

    obj.generate_dataset()

    if args.gpu:
        logging.info("try to use cuda")
        obj.device = torch.device("cuda")
        obj.model.device = obj.device

    if args.train:
        obj.train_()
        obj.save(args.modelpath)

    if args.test:
        # obj.test()
        obj.test_by_nag_pos()
    pass
