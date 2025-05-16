import os
import sys
import contextlib
import logging
import torch
import numpy as np
from models import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def clamp(X, lower_limit, upper_limit):
    if isinstance(lower_limit, float) and isinstance(upper_limit, float):
        return X.clamp(min=lower_limit, max=upper_limit)
    else:
        return torch.max(torch.min(X, upper_limit), lower_limit)


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_model(model_name):
    if not model_name.endswith(")"):
        model_name += "()"
    return eval(model_name)


def get_optimizer(args, model):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    if args.lr_type == 'cyclic':
        lr_schedule = lambda t: np.interp(
            [t], [0, args.epochs * 2//5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat':
        lr_schedule = lambda t: args.lr_max
    elif args.lr_type == 'decay':
        lr_schedule = lambda t: args.lr_max - t * args.lr_max / args.epochs
    else:
        raise ValueError('Unknown lr_type')
    return opt, lr_schedule
