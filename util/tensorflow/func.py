__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["inner_product", "euclidean_distance", "l2_distance",
           "get_variable"]

import tensorflow as tf
from reckit import typeassert
from collections import OrderedDict
from util.common import InitArg


@typeassert(init_method=str, trainable=bool, name=(str, None))
def get_variable(shape, init_method, trainable=True, name=None):
    initializers = OrderedDict()
    initializers["normal"] = tf.initializers.random_normal(mean=InitArg.MEAN, stddev=InitArg.STDDEV)
    initializers["truncated_normal"] = tf.initializers.truncated_normal(mean=InitArg.MEAN, stddev=InitArg.STDDEV)
    initializers["uniform"] = tf.initializers.random_uniform(minval=InitArg.MIN_VAL, maxval=InitArg.MAX_VAL)
    initializers["he_normal"] = tf.initializers.he_normal()
    initializers["he_uniform"] = tf.initializers.he_uniform()
    initializers["xavier_normal"] = tf.initializers.glorot_normal()
    initializers["xavier_uniform"] = tf.initializers.glorot_uniform()
    initializers["zeros"] = tf.initializers.zeros()
    initializers["ones"] = tf.initializers.ones()

    if init_method not in initializers:
        init_list = ', '.join(initializers.keys())
        raise ValueError(f"'init_method' is invalid, and must be one of '{init_list}'")
    init = initializers[init_method](shape=shape, dtype=tf.float32)

    return tf.Variable(init, trainable=trainable, name=name)


def inner_product(a, b):
    return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def euclidean_distance(a, b):
    return tf.norm(a - b, ord='euclidean', axis=-1)


l2_distance = euclidean_distance
