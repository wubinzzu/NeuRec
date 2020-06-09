# distutils: language = c++
"""
@author: Zhongchuan Sun
"""
import numpy as np
cimport numpy as np
import os
from .tools import float_type, is_ndarray, int_type


cdef extern from "arg_topk.h":
    void arg_top_k_2d(float *scores_pt, int columns_num, int rows_num,
                      int top_k, int thread_num, int *results_pt)


def arg_topk(ranking_scores, top_k=50, thread_num=None):
    users_num, rank_len = np.shape(ranking_scores)

    thread_num = (thread_num or (os.cpu_count() or 1) * 5)


    if not is_ndarray(ranking_scores, float_type):
        ranking_scores = np.array(ranking_scores, dtype=float_type)

    # get the pointer of ranking scores
    cdef float *scores_pt = <float *>np.PyArray_DATA(ranking_scores)

    # store ranks results
    topk_indices = np.zeros([users_num, top_k], dtype=int_type)
    cdef int *topk_indices_pt = <int *>np.PyArray_DATA(topk_indices)

    # get top k rating index
    arg_top_k_2d(scores_pt, rank_len, users_num, top_k, thread_num, topk_indices_pt)

    return topk_indices
