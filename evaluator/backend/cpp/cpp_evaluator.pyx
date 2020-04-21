# distutils: language = c++
"""
@author: Zhongchuan Sun
"""
import numpy as np
cimport numpy as np
from evaluator.abstract_evaluator import AbstractEvaluator
from .apt_tools import float_type
from libcpp.unordered_map cimport unordered_map as cmap
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.vector cimport vector as cvector


ctypedef cset[int] int_set

cdef extern from "include/evaluate.h":
    void cpp_evaluate_matrix(float *rating_matrix, int rating_len,
                             cvector[int] &test_users,
                             cmap[int, int_set] &all_truth,
                             cvector[int] metric, int top_k,
                             int thread_num, float *results_pt)


class CPPEvaluator(AbstractEvaluator):
    """Evaluator for item ranking task.
    """
    def __init__(self, user_test_dict):
        super(CPPEvaluator, self).__init__()

        cdef cmap[int, int_set] user_pos_cmap
        cdef int_set item_set
        for user, items in user_test_dict.items():
            item_set = items
            user_pos_cmap[user] = item_set
        self.user_pos_cmap = user_pos_cmap

    def eval_score_matrix(self, score_matrix, test_users, metric, top_k, thread_num):
        rating_len = np.shape(score_matrix)[-1]
        cdef float *scores_pt = <float *>np.PyArray_DATA(score_matrix)
        cdef cvector[int] test_users_vec = test_users
        cdef cvector[int] metric_vec = metric

        # evaluation results
        user_num = len(test_users)
        metrics_num = len(metric)
        results = np.zeros([user_num, metrics_num*top_k], dtype=float_type)
        results_pt = <float *>np.PyArray_DATA(results)
        cpp_evaluate_matrix(scores_pt, rating_len, test_users_vec, self.user_pos_cmap,
                            metric_vec, top_k, thread_num, results_pt)

        return results
