from typing import Mapping
from loguru import logger
from torch import Tensor


def shapes(**t: Tensor):
    for k, v in t.items():
        logger.opt(depth=1).debug(f'shape of {k}: {v.shape}')


def minmax(**t: Tensor):
    for k, v in t.items():
        logger.opt(depth=1).debug(f'min and max of {k}: {v.min()}, {v.max()}')
