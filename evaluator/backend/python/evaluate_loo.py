"""
@author: Zhongchuan Sun
"""
import numpy as np
from util import typeassert, argmax_top_k
from concurrent.futures import ThreadPoolExecutor
import sys


def hit(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item == ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0
    return result


def ndcg(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item == ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0/np.log2(last_idx+2)
    return result


def mrr(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item == ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0/(last_idx+1)
    return result


@typeassert(score_matrix=np.ndarray, test_items=(list, np.ndarray), top_k=int)
def eval_score_matrix_loo(score_matrix, test_items, top_k=50, thread_num=None):
    def _eval_one_user(idx):
        scores = score_matrix[idx]  # all scores of the test user
        test_item = test_items[idx]  # all test items of the test user

        ranking = argmax_top_k(scores, top_k)  # Top-K items
        result = []
        result.extend(hit(ranking, test_item))
        result.extend(ndcg(ranking, test_item))
        result.extend(mrr(ranking, test_item))

        result = np.array(result, dtype=np.float32).flatten()
        return result

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        batch_result = executor.map(_eval_one_user, range(len(test_items)))

    result = list(batch_result)  # generator to list
    return np.array(result)  # list to ndarray
