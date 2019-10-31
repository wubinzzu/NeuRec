# distutils: language = c++
"""
@author: Zhongchuan Sun
"""
import numpy as np
cimport numpy as np
import os
from .apt_tools import get_float_type, get_int_type, is_ndarray
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef extern from "include/tools.h":
    void c_top_k_array_index(float *scores_pt, int columns_num, int rows_num,
                             int top_k, int thread_num, int *rankings_pt)

cdef extern from "include/evaluate_foldout.h":
    void evaluate_foldout(int users_num,
                          int *rankings, int rank_len,
                          int **ground_truths, int *ground_truths_num,
                          int thread_num, float *results)


def apt_evaluate_foldout(ranking_scores, ground_truth, top_k=50, thread_num=None):
    metrics_num = 5
    users_num, rank_len = np.shape(ranking_scores)
    if users_num != len(ground_truth):
        raise Exception("The lengths of 'ranking_scores' and 'ground_truth' are different.")
    thread_num = (thread_num or (os.cpu_count() or 1) * 5)

    float_type = get_float_type()
    int_type = get_int_type()

    if not is_ndarray(ranking_scores, float_type):
        ranking_scores = np.array(ranking_scores, dtype=float_type)

    # get the pointer of ranking scores
    cdef float *scores_pt = <float *>np.PyArray_DATA(ranking_scores)

    # store ranks results
    top_rankings = np.zeros([users_num, top_k], dtype=int_type)
    cdef int *rankings_pt = <int *>np.PyArray_DATA(top_rankings)

    # get top k rating index
    c_top_k_array_index(scores_pt, rank_len, users_num, top_k, thread_num, rankings_pt)

    # the pointer of ground truth, the pointer of the length array of ground truth
    ground_truth_pt = <int **> PyMem_Malloc(users_num * sizeof(int *))
    ground_truth_num = np.zeros([users_num], dtype=int_type)
    ground_truth_num_pt = <int *>np.PyArray_DATA(ground_truth_num)
    for u in range(users_num):
        if not is_ndarray(ground_truth[u], int_type):
            ground_truth[u] = np.array(ground_truth[u], dtype=int_type, copy=True)
        ground_truth_pt[u] = <int *>np.PyArray_DATA(ground_truth[u])
        ground_truth_num[u] = len(ground_truth[u])

    #evaluate results
    results = np.zeros([users_num, metrics_num*top_k], dtype=float_type)
    results_pt = <float *>np.PyArray_DATA(results)

    #evaluate
    evaluate_foldout(users_num, rankings_pt, top_k, ground_truth_pt, ground_truth_num_pt, thread_num, results_pt)

    #release the allocated space
    PyMem_Free(ground_truth_pt)

    return results
