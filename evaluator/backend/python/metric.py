"""
@author: Zhongchuan Sun
"""
import numpy as np
import sys


def hit(rank, ground_truth):
    # HR is equal to Recall when dataset is loo split.
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item == ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0
    return result


def precision(rank, ground_truth):
    # Precision is meaningless when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32)/np.arange(1, len(rank)+1)
    return result


def recall(rank, ground_truth):
    # Recall is equal to HR when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / len(ground_truth)
    return result


def map(rank, ground_truth):
    pre = precision(rank, ground_truth)
    pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    # relevant_num = np.cumsum([1 if item in ground_truth else 0 for item in rank])
    relevant_num = np.cumsum([min(idx+1, len(ground_truth)) for idx, _ in enumerate(rank)])
    result = [p/r_num if r_num!=0 else 0 for p, r_num in zip(sum_pre, relevant_num)]
    return result


def ndcg(rank, ground_truth):
    len_rank = len(rank)
    idcg_len = min(len(ground_truth), len_rank)
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len - 1]

    dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    result = dcg/idcg
    return result


def mrr(rank, ground_truth):
    # MRR is equal to MAP when dataset is loo split.
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item in ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0/(last_idx+1)
    return result


metric_dict = {"Precision": precision,
               "Recall": recall,
               "MAP": map,
               "NDCG": ndcg,
               "MRR": mrr}
