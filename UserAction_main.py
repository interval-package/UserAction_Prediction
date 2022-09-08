import logging
import argparse

from UserAction_Run import *


parser = argparse.ArgumentParser()
parser.add_argument()
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("--path", action="path",
                    default='./runs/logs/model.log',
                    help="log saving path")

if __name__ == '__main__':

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO if args.verbose else logging.WARNING,
                        filename=args.path,
                        filemode='a')
    obj = UserAction_run()
    obj.train()
    obj.save()
    obj.test()
    pass
