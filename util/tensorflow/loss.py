__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"


__all__ = ["l2_loss",
           "square_loss", "sigmoid_cross_entropy", "pointwise_loss",
           "bpr_loss", "hinge", "pairwise_loss"]

import tensorflow as tf
from reckit import typeassert
from collections import OrderedDict
from util.common import Reduction
from functools import partial


def _reduce_loss(loss, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    if reduction == Reduction.SUM:
        loss = tf.reduce_sum(loss)
    elif reduction == Reduction.MEAN:
        loss = tf.reduce_mean(loss)
    elif reduction == Reduction.NONE:
        pass

    return loss


def square_loss(y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    loss = tf.squared_difference(y_pre, y_true)
    return _reduce_loss(loss, reduction)


def sigmoid_cross_entropy(y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pre)
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


def bpr_loss(y_diff, reduction=Reduction.SUM):
    """bpr loss
    """
    Reduction.validate(reduction)

    loss = -tf.log_sigmoid(y_diff)
    return _reduce_loss(loss, reduction)


def hinge(y_diff, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    loss = tf.nn.relu(1.0-y_diff)
    return _reduce_loss(loss, reduction)


@typeassert(loss=str, reduction=str)
def pairwise_loss(loss, y_diff, reduction=Reduction.SUM):
    Reduction.validate(reduction)

    losses = OrderedDict()
    losses["bpr"] = bpr_loss
    losses["hinge"] = hinge
    losses["square"] = partial(square_loss, y_true=1.0)

    if loss not in losses:
        loss_list = ', '.join(losses.keys())
        ValueError(f"'loss' is invalid, and must be one of '{loss_list}'")

    return losses[loss](y_diff, reduction=reduction)


def l2_loss(*weights):
    """L2 loss

    Compute  the L2 norm of tensors without the `sqrt`:

        output = sum([sum(w ** 2) / 2 for w in weights])

    Args:
        *weights: Variable length weight list.

    """
    return tf.add_n([tf.nn.l2_loss(w) for w in weights])
