# distutils: language = c++
"""
@author: Zhongchuan Sun
"""
import numpy as np
cimport numpy as np
from evaluator.abstract_evaluator import AbstractEvaluator
from util.cython.tools import float_type
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.vector cimport vector as cvector


ctypedef cset[int] int_set

cdef extern from "include/evaluate.h":
    void cpp_evaluate_matrix(float *rating_matrix, int rating_len,
                             cvector[int_set] &test_items,
                             cvector[int] metric, int top_k,
                             int thread_num, float *results_pt)


class CPPEvaluator(AbstractEvaluator):
    """Evaluator for item ranking task.
    """
    def __init__(self):
        super(CPPEvaluator, self).__init__()

    def eval_score_matrix(self, score_matrix, test_items, metric, top_k, thread_num):
        rating_len = np.shape(score_matrix)[-1]
        user_num = len(test_items)
        cdef float *scores_pt = <float *>np.PyArray_DATA(score_matrix)
        cdef cvector[int_set] test_items_vec = test_items
        cdef cvector[int] metric_vec = metric

        # evaluation results
        metrics_num = len(metric)
        results = np.zeros([user_num, metrics_num*top_k], dtype=float_type)
        results_pt = <float *>np.PyArray_DATA(results)
        cpp_evaluate_matrix(scores_pt, rating_len, test_items_vec,
                            metric_vec, top_k, thread_num, results_pt)

        return results
