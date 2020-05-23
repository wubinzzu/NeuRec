import tensorflow as tf
import numpy as np
from inspect import signature
from functools import wraps
import heapq
import itertools
import time
from concurrent.futures import ThreadPoolExecutor


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
            
        elif act == "softmax":
            act_func = tf.nn.softmax(act_input)
         
        elif act == "selu":
            act_func = tf.nn.selu(act_input) 
        
        else:
            raise NotImplementedError("ERROR")
        return act_func  


def get_data_format(data_format):
    if data_format == "UIRT":
        columns = ["user", "item", "rating", "time"]
        
    elif data_format == "UIR":
        columns = ["user", "item", "rating"]
    
    elif data_format == "UIT":
        columns = ["user", "item", "time"] 
        
    elif data_format == "UI":
        columns = ["user", "item"]    
    
    else:
        raise ValueError("please choose a correct data format. ")
    
    return columns


def csr_to_user_dict(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        if any(value.indices):
            train_dict[idx] = value.indices.copy().tolist()
    return train_dict


def csr_to_user_dict_bytime(time_matrix,train_matrix):
    train_dict = {}
    time_matrix = time_matrix
    user_pos_items = csr_to_user_dict(train_matrix)
    for u, items in user_pos_items.items():
        sorted_items = sorted(items, key=lambda x: time_matrix[u,x])
        train_dict[u] = np.array(sorted_items, dtype=np.int32).tolist()

    return train_dict


def get_initializer(init_method, stddev):
        if init_method == 'tnormal':
            return tf.truncated_normal_initializer(stddev=stddev)
        elif init_method == 'uniform':
            return tf.random_uniform_initializer(-stddev, stddev)
        elif init_method == 'normal':
            return tf.random_normal_initializer(stddev=stddev)
        elif init_method == 'xavier_normal':
            return tf.contrib.layers.xavier_initializer(uniform=False)
        elif init_method == 'xavier_uniform':
            return tf.contrib.layers.xavier_initializer(uniform=True)
        elif init_method == 'he_normal':
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)
        elif init_method == 'he_uniform':
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=True)
        else:
            return tf.truncated_normal_initializer(stddev=stddev)  


def noise_validator(noise, allowed_noises):
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass 


def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).
    """
    a = np.arange(high)
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = p.flatten()
        p[exclusion] = 0
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample


def batch_random_choice(high, size, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).
    :param high: integer
    :param size: 1-D array_like
    :param replace: bool
    :param p: 2-D array_like
    :param exclusion: a list of 1-D array_like
    :return: a list of 1-D array_like sample
    """

    if p is not None and (len(p) != len(size) or len(p[0]) != high):
        raise ValueError("The shape of 'p' is not compatible with the shapes of 'array' and 'size'!")

    if exclusion is not None and len(exclusion) != len(size):
        raise ValueError("The shape of 'exclusion' is not compatible with the shape of 'size'!")

    def choice_one(idx):
        p_tmp = p[idx] if p is not None else None
        exc = exclusion[idx] if exclusion is not None else None
        return randint_choice(high, size[idx], replace=replace, p=p_tmp, exclusion=exc)

    with ThreadPoolExecutor() as executor:
        results = executor.map(choice_one, range(len(size)))

    return [result for result in results]


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


def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)


def pad_sequences(sequences, value=0., max_len=None,
                  padding='post', truncating='post', dtype=np.int32):
    """Pads sequences to the same length.

    Args:
        sequences (list): A list of lists, where each element is a sequence.
        value (int or float): Padding value. Defaults to `0.`.
        max_len (int or None): Maximum length of all sequences.
        padding (str): `"pre"` or `"post"`: pad either before or after each
            sequence. Defaults to `post`.
        truncating (str): `"pre"` or `"post"`: remove values from sequences
            larger than `max_len`, either at the beginning or at the end of
            the sequences. Defaults to `post`.
        dtype (int or float): Type of the output sequences. Defaults to `np.int32`.

    Returns:
        np.ndarray: Numpy array with shape `(len(sequences), max_len)`.

    Raises:
        ValueError: If `padding` or `truncating` is not understood.
    """
    if max_len is None:
        max_len = np.max([len(x) for x in sequences])

    x = np.full([len(sequences), max_len], value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def inner_product(a, b, name="inner_product"):
    with tf.name_scope(name=name):
        return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def timer(func):
    """The timer decorator
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("%s function cost: %fs" % (func.__name__, end_time - start_time))
        return result
    return wrapper


def l2_loss(*params):
    return tf.add_n([tf.nn.l2_loss(w) for w in params])


def log_loss(yij, name="log_loss"):
    """ bpr loss
    """
    with tf.name_scope(name):
        return -tf.log_sigmoid(yij)
