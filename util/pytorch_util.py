__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["inner_product", "log_loss", "l2_loss"]

import torch
import torch.nn.functional as F


def inner_product(a, b):
    return torch.sum(a*b, dim=-1)


def log_loss(yij):
    """bpr loss
    """
    return F.softplus(-yij)


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
