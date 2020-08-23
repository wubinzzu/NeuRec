__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["inner_product", "euclidean_distance", "l2_distance",
           "get_initializer",
           "sp_mat_to_sp_tensor", "dropout_sparse"]

import torch
from torch import nn
from functools import partial
from collections import OrderedDict
from reckit import typeassert
from util.common import InitArg
import numpy as np


def truncated_normal_(tensor, mean=0.0, std=1.0):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_(mean=0, std=1)
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


_initializers = OrderedDict()
_initializers["normal"] = partial(nn.init.normal_, mean=InitArg.MEAN, std=InitArg.STDDEV)
_initializers["truncated_normal"] = partial(truncated_normal_, mean=InitArg.MEAN, std=InitArg.STDDEV)
_initializers["uniform"] = partial(nn.init.uniform_, a=InitArg.MIN_VAL, b=InitArg.MAX_VAL)
_initializers["he_normal"] = nn.init.kaiming_normal_
_initializers["he_uniform"] = nn.init.kaiming_uniform_
_initializers["xavier_normal"] = nn.init.xavier_normal_
_initializers["xavier_uniform"] = nn.init.xavier_uniform_
_initializers["zeros"] = nn.init.zeros_
_initializers["ones"] = nn.init.ones_


@typeassert(init_method=str)
def get_initializer(init_method):
    if init_method not in _initializers:
        init_list = ', '.join(_initializers.keys())
        raise ValueError(f"'init_method' is invalid, and must be one of '{init_list}'")
    return _initializers[init_method]


def inner_product(a, b):
    return torch.sum(a*b, dim=-1)


def euclidean_distance(a, b):

    return torch.norm(a-b, p=None, dim=-1)


l2_distance = euclidean_distance


def sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()


def dropout_sparse(torch_sp_mat, keep_prob, training):
    """Dropout for sparse tensors.
    """
    if keep_prob <= 0.0 or keep_prob > 1.0:
        raise ValueError(f"'keep_prob' must be a float in the range (0, 1], got {keep_prob}")
    if training:
        device = torch_sp_mat.device
        values = torch_sp_mat.values()
        noise_shape = values.shape

        random_tensor = torch.Tensor(noise_shape).uniform_().to(device) + keep_prob
        dropout_mask = random_tensor.floor().bool()

        indices = torch_sp_mat.indices()
        indices = indices[:, dropout_mask]
        scale = 1.0 / keep_prob
        values = values[dropout_mask]*scale
        shape = torch_sp_mat.shape

        torch_sp_mat = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)

    return torch_sp_mat
