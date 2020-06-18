# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
"""
@author: Zhongchuan Sun
"""
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.vector cimport vector as cvector
from libc.stdlib cimport rand, srand
ctypedef cset[int] int_set


cdef llrand():
    cdef unsigned long long r = 0
    cdef int i = 0
    for i in range(5):
        r = (r << 15) | (rand() & 0x7FFF)
    return r & 0xFFFFFFFFFFFFFFFFULL


def randint_choice(high, size=1, replace=True, p=None, exclusion=None):
    """Sample random integers from [0, high).
    """
    if size <= 0:
        raise ValueError("'size' must be a positive integer.")

    if not isinstance(replace, bool):
        raise TypeError("'replace' must be bool.")

    if p is not None:
        raise NotImplementedError

    if exclusion is not None and high <= len(exclusion):
        raise ValueError("The number of 'exclusion' is greater than 'high'.")

    len_exclusion = len(exclusion) if exclusion is not None else 0
    if replace is False and (high-len_exclusion <= size):
        raise ValueError("There is not enough integers to be sampled.")

    cdef int_set omission
    if exclusion is not None:
        for elem in exclusion:
            omission.insert(elem)

    cdef cvector[int] c_arr
    cdef int a
    cdef int i = 0
    cdef int c_high = high
    cdef int c_replace = replace
    cdef int c_size = size
    while c_size - i:
        a = llrand() % c_high
        if not omission.count(a):
            c_arr.push_back(a)
            i += 1
            if not c_replace:
                omission.insert(a)

    if size == 1:
        tmp = c_arr[0]
    else:
        tmp = c_arr
    return tmp

def batch_randint_choice(high, size, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).

    Args:
        high (int):
        size: 1-D array_like
        replace (bool):
        p: 2-D array_like
        exclusion: a list of 1-D array_like

    Returns:
        list: a list of 1-D array_like sample

    """
    if p is not None:
        raise NotImplementedError

    if exclusion is not None and len(size) != len(exclusion):
        raise ValueError("The shape of 'exclusion' is not compatible with the shape of 'size'!")

    results = []
    for idx in range(len(size)):
        p_tmp = p[idx] if p is not None else None
        exc = exclusion[idx] if exclusion is not None else None
        results.append(randint_choice(high, size=size[idx], replace=replace, p=p_tmp, exclusion=exc))
    return results
