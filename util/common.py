__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Reduction", "InitArg", "normalize_adj_matrix"]

import numpy as np
import scipy.sparse as sp


class Reduction(object):
    NONE = "none"
    SUM = "sum"
    MEAN = "mean"

    @classmethod
    def all(cls):
        return (cls.NONE,
                cls.SUM,
                cls.MEAN)

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            key_list = ', '.join(cls.all())
            raise ValueError(f"{key} is an invalid Reduction Key, which must be one of '{key_list}'.")


class InitArg(object):
    MEAN = 0.0
    STDDEV = 0.01
    MIN_VAL = -0.05
    MAX_VAL = 0.05


def normalize_adj_matrix(sp_mat, norm_method="left"):
    """Normalize adjacent matrix

    Args:
        sp_mat: A sparse adjacent matrix
        norm_method (str): The normalization method, can be 'symmetric'
            or 'left'.

    Returns:
        sp.spmatrix: The normalized adjacent matrix.

    """

    d_in = np.asarray(sp_mat.sum(axis=1))  # indegree
    if norm_method == "left":
        rec_d_in = np.power(d_in, -1).flatten()  # reciprocal
        rec_d_in[np.isinf(rec_d_in)] = 0.  # replace inf
        rec_d_in = sp.diags(rec_d_in)  # to diagonal matrix
        norm_sp_mat = rec_d_in.dot(sp_mat)  # left matmul
    elif norm_method == "symmetric":
        rec_sqrt_d_in = np.power(d_in, -0.5).flatten()
        rec_sqrt_d_in[np.isinf(rec_sqrt_d_in)] = 0.
        rec_sqrt_d_in = sp.diags(rec_sqrt_d_in)

        mid_sp_mat = rec_sqrt_d_in.dot(sp_mat)  # left matmul
        norm_sp_mat = mid_sp_mat.dot(rec_sqrt_d_in)  # right matmul
    else:
        raise ValueError(f"'{norm_method}' is an invalid normalization method.")

    return norm_sp_mat
