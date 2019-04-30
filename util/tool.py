import tensorflow as tf
import time
import numpy as np
from inspect import signature
from functools import wraps
from scipy.sparse import csr_matrix


def activation_function(act,act_input):
    act_func = None
    if act == "sigmoid":
        act_func = tf.nn.sigmoid(act_input)
    elif act == "tanh":
        act_func = tf.nn.tanh(act_input)

    elif act == "relu":
        act_func = tf.nn.relu(act_input)

    elif act == "elu":
        act_func = tf.nn.elu(act_input)

    elif act == "identity":
        act_func = tf.identity(act_input)
    else:
        raise NotImplementedError("ERROR")
    return act_func


def getlocaltime():
    date = time.strftime('%y-%m-%d', time.localtime())
    current_time = time.strftime('%H:%M:%S', time.localtime())


def typeassert(*type_args, **type_kwargs):
    def decorate(func):
        sig = signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate


def timer(func):
    """The timer decorator
    """
    def inner(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("%s function cost: %fs" % (func.__name__, end_time - start_time))
        return result
    return inner


def random_choice(a, size=None, replace=True, p=None, exclusion=None):
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = np.ndarray.flatten(p)
        p[exclusion] = 0
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample


@typeassert(sparse_matrix_data=csr_matrix)
def csr_to_user_dict(sparse_matrix_data):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    idx_value_dict = {}
    for idx, value in enumerate(sparse_matrix_data):
        if any(value.indices):
            idx_value_dict[idx] = value.indices
    return idx_value_dict
