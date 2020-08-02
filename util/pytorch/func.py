__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["inner_product", "euclidean_distance", "l2_distance",
           "init_variable"]

import torch
from torch import nn
from functools import partial
from collections import OrderedDict
from reckit import typeassert
from util.common import InitArg


@typeassert(init_method=str)
def init_variable(varibale, init_method):
    initializers = OrderedDict()
    initializers["normal"] = partial(nn.init.normal_, mean=InitArg.MEAN, std=InitArg.STDDEV)
    initializers["truncated_normal"] = partial(truncated_normal_, mean=InitArg.MEAN, std=InitArg.STDDEV)
    initializers["uniform"] = partial(nn.init.uniform_, a=InitArg.MIN_VAL, b=InitArg.MAX_VAL)
    initializers["he_normal"] = nn.init.kaiming_normal_
    initializers["he_uniform"] = nn.init.kaiming_uniform_
    initializers["xavier_normal"] = nn.init.xavier_normal_
    initializers["xavier_uniform"] = nn.init.xavier_uniform_
    initializers["zeros"] = nn.init.zeros_
    initializers["ones"] = nn.init.ones_

    if init_method not in initializers:
        init_list = ', '.join(initializers.keys())
        raise ValueError(f"'init_method' is invalid, and must be one of '{init_list}'")

    initializers[init_method](varibale)


def truncated_normal_(tensor, mean=0.0, std=1.0):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_(mean=0, std=1)
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def inner_product(a, b):
    return torch.sum(a*b, dim=-1)


def euclidean_distance(a, b):

    return torch.norm(a-b, p=None, dim=-1)


l2_distance = euclidean_distance
