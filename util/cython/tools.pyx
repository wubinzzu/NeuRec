# distutils: language = c++
"""
@author: Zhongchuan Sun
"""
import numpy as np

def get_float_type():
    cdef size_of_float = sizeof(float)*8
    if size_of_float == 32:
        return np.float32
    elif size_of_float == 64:
        return np.float64
    else:
        raise EnvironmentError("The size of 'float' is %d, but 32 or 64." % size_of_float)

def get_int_type():
    cdef size_of_int = sizeof(int)*8
    if size_of_int == 16:
        return np.int16
    elif size_of_int == 32:
        return np.int32
    else:
        raise EnvironmentError("The size of 'int' is %d, but 16 or 32." % size_of_int)


float_type = get_float_type()
int_type = get_int_type()


def is_ndarray(array, dtype):
    if not isinstance(array, np.ndarray):
        return False
    if array.dtype != dtype:
        return False
    if array.base is not None:
        return False
    return True
