__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["inner_product", "euclidean_distance", "l2_distance",
           "get_initializer"]

import tensorflow as tf
from reckit import typeassert
from collections import OrderedDict
from util.common import InitArg

_initializers = OrderedDict()
_initializers["normal"] = tf.initializers.random_normal(mean=InitArg.MEAN, stddev=InitArg.STDDEV)
_initializers["truncated_normal"] = tf.initializers.truncated_normal(mean=InitArg.MEAN, stddev=InitArg.STDDEV)
_initializers["uniform"] = tf.initializers.random_uniform(minval=InitArg.MIN_VAL, maxval=InitArg.MAX_VAL)
_initializers["he_normal"] = tf.initializers.he_normal()
_initializers["he_uniform"] = tf.initializers.he_uniform()
_initializers["xavier_normal"] = tf.initializers.glorot_normal()
_initializers["xavier_uniform"] = tf.initializers.glorot_uniform()
_initializers["zeros"] = tf.initializers.zeros()
_initializers["ones"] = tf.initializers.ones()


@typeassert(init_method=str)
def get_initializer(init_method):
    if init_method not in _initializers:
        init_list = ', '.join(_initializers.keys())
        raise ValueError(f"'init_method' is invalid, and must be one of '{init_list}'")
    return _initializers[init_method]


def inner_product(a, b):
    return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def euclidean_distance(a, b):
    return tf.norm(a - b, ord='euclidean', axis=-1)


l2_distance = euclidean_distance
