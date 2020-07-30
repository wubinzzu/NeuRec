__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["inner_product", "l2_loss", "init_variable",
           "square_loss", "sigmoid_cross_entropy", "pointwise_loss",
           "log_sigmoid", "bpr_loss", "hinge", "pairwise_loss"]

import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from collections import OrderedDict
from reckit import typeassert
from util.common_util import InitArg
from util.common_util import Reduction


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


def _reduce_loss(loss, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    if reduction == Reduction.SUM:
        loss = torch.sum(loss)
    elif reduction == Reduction.MEAN:
        loss = torch.mean(loss)
    elif reduction == Reduction.NONE:
        pass

    return loss


def square_loss(y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    if isinstance(y_true, (float, int)):
        y_true = y_pre.new_full(y_pre.size(), y_true)

    loss = F.mse_loss(input=y_pre, target=y_true, reduce=False)
    return _reduce_loss(loss, reduction)


def sigmoid_cross_entropy(y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    y_pre = F.sigmoid(y_pre)
    loss = F.binary_cross_entropy(input=y_pre, target=y_true, reduce=False)
    return _reduce_loss(loss, reduction)


@typeassert(loss=str, reduction=str)
def pointwise_loss(loss, y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)

    losses = OrderedDict()
    losses["square"] = square_loss
    losses["sigmoid_cross_entropy"] = sigmoid_cross_entropy

    if loss not in losses:
        loss_list = ', '.join(losses.keys())
        ValueError(f"'loss' is invalid, and must be one of '{loss_list}'")

    return losses[loss](y_pre, y_true, reduction=reduction)


def log_sigmoid(y_diff, reduction=Reduction.SUM):
    """bpr loss
    """
    Reduction.validate(reduction)
    loss = F.softplus(-y_diff)
    return _reduce_loss(loss, reduction)


bpr_loss = log_sigmoid


def hinge(y_diff, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    ones = y_diff.new_ones(y_diff.size())
    loss = torch.relu(ones-y_diff)
    return _reduce_loss(loss, reduction)


@typeassert(loss=str, reduction=str)
def pairwise_loss(loss, y_diff, reduction=Reduction.SUM):
    Reduction.validate(reduction)

    losses = OrderedDict()
    losses["log_sigmoid"] = log_sigmoid
    losses["bpr"] = bpr_loss
    losses["hinge"] = hinge
    losses["square"] = partial(square_loss, y_true=1.0)

    if loss not in losses:
        loss_list = ', '.join(losses.keys())
        ValueError(f"'loss' is invalid, and must be one of '{loss_list}'")

    return losses[loss](y_diff, reduction=reduction)


def inner_product(a, b):
    return torch.sum(a*b, dim=-1)


def l2_loss(*weights):
    """L2 loss

    Compute  the L2 norm of tensors without the `sqrt`:

        output = sum([sum(w ** 2) / 2 for w in weights])

    Args:
        *weights: Variable length weight list.

    """
    loss = 0.0
    for w in weights:
        loss += torch.sum(torch.pow(w, 2))

    return 0.5*loss
