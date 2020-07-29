__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["inner_product", "l2_loss", "log_loss"]

import tensorflow as tf


def inner_product(a, b, name="inner_product"):
    with tf.name_scope(name=name):
        return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def l2_loss(*params):
    return tf.add_n([tf.nn.l2_loss(w) for w in params])


def log_loss(yij, name="log_loss"):
    """bpr loss
    """
    with tf.name_scope(name):
        return -tf.log_sigmoid(yij)
