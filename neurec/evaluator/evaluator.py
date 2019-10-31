"""
@author: Zhongchuan Sun
"""
from neurec.util.tool import csr_to_user_dict
from scipy.sparse import csr_matrix
from neurec.util.tool import typeassert, pad_sequences
from neurec.util.DataIterator import DataIterator
import numpy as np
from neurec.evaluator.backend import eval_score_matrix_foldout, eval_score_matrix_loo

class AbstractEvaluator(object):
    """Basic class for evaluator.
    """
    def __init__(self):
        pass

    def metrics_info(self):
        raise NotImplementedError

    def evaluate(self, ranking_score):
        raise NotImplementedError


class FoldOutEvaluator(AbstractEvaluator):
    """Evaluator for generic ranking task.
    """
    @typeassert(train_matrix=csr_matrix, test_matrix=csr_matrix, top_k=(int, list, tuple, np.ndarray))
    def __init__(self, train_matrix, test_matrix, negative_matrix=None, top_k=50):
        super(FoldOutEvaluator, self).__init__()
        self.max_top = top_k if isinstance(top_k, int) else int(max(top_k))
        if isinstance(top_k, int):
            self.top_show = np.arange(top_k)+1
        else:
            self.top_show = np.sort(top_k)
        self.user_pos_train = csr_to_user_dict(train_matrix)
        self.user_pos_test = csr_to_user_dict(test_matrix)
        self.user_neg_test = csr_to_user_dict(negative_matrix) if negative_matrix is not None else None
        self.metrics_num = 5

    def metrics_info(self):
        Precision = '\t'.join(["Precision@"+str(k) for k in self.top_show])
        Recall = '\t'.join(["Recall@" + str(k) for k in self.top_show])
        MAP = '\t'.join(["MAP@" + str(k) for k in self.top_show])
        NDCG = '\t'.join(["NDCG@" + str(k) for k in self.top_show])
        MRR = '\t'.join(["MRR@" + str(k) for k in self.top_show])
        mertic = '\t'.join([Precision, Recall, MAP, NDCG, MRR])
        return mertic

    def evaluate(self, model):
        # B: batch size
        # N: the number of items
        test_users = DataIterator(list(self.user_pos_test.keys()), batch_size=2048, shuffle=False, drop_last=False)
        batch_result = []
        for batch_users in test_users:
            if self.user_neg_test is not None:
                candidate_items = []
                test_items = []
                for user in batch_users:
                    pos = self.user_pos_test[user]
                    neg = self.user_neg_test[user]
                    candidate_items.append(pos+neg)
                    test_items.append(list(range(len(pos))))
                ranking_score = model.predict(batch_users, candidate_items)  # (B,N)
                ranking_score = pad_sequences(ranking_score, value=-np.inf)

                ranking_score = np.array(ranking_score)
            else:
                test_items = []
                for user in batch_users:
                    test_items.append(self.user_pos_test[user])
                ranking_score = model.predict(batch_users, None)  # (B,N)
                ranking_score = np.array(ranking_score)

                # set the ranking scores of training items to -inf,
                # then the training items will be sorted at the end of the ranking list.
                for idx, user in enumerate(batch_users):
                    train_items = self.user_pos_train[user]
                    ranking_score[idx][train_items] = -np.inf

            result = eval_score_matrix_foldout(ranking_score, test_items, top_k=self.max_top, thread_num=None)  # (B,k*metric_num)
            batch_result.append(result)

        # concatenate the batch results to a matrix
        all_user_result = np.concatenate(batch_result, axis=0)
        final_result = np.mean(all_user_result, axis=0)  # mean

        final_result = np.reshape(final_result, newshape=[self.metrics_num, self.max_top])
        final_result = final_result[:, self.top_show-1]
        final_result = np.reshape(final_result, newshape=[-1])
        buf = '\t'.join(["%.8f" % x for x in final_result])
        return buf


class LeaveOneOutEvaluator(AbstractEvaluator):
    """Evaluator for leave one out ranking task.
    """
    @typeassert(train_matrix=csr_matrix, test_matrix=csr_matrix, top_k=(int, list, tuple, np.ndarray))
    def __init__(self, train_matrix, test_matrix, negative_matrix=None, top_k=50):
        super(LeaveOneOutEvaluator, self).__init__()
        self.max_top = top_k if isinstance(top_k, int) else int(max(top_k))
        if isinstance(top_k, int):
            self.top_show = np.arange(top_k)+1
        else:
            self.top_show = np.sort(top_k)
        self.user_pos_train = csr_to_user_dict(train_matrix)
        self.user_pos_test = csr_to_user_dict(test_matrix)
        self.user_neg_test = csr_to_user_dict(negative_matrix) if negative_matrix is not None else None
        self.metrics_num = 3

    def metrics_info(self):
        HR = '\t'.join(["HR@"+str(k) for k in self.top_show])
        NDCG = '\t'.join(["NDCG@" + str(k) for k in self.top_show])
        MRR = '\t'.join(["MRR@" + str(k) for k in self.top_show])
        mertic = '\t'.join([HR, NDCG, MRR])
        return mertic

    def evaluate(self, model):
        # B: batch size
        # N: the number of items
        test_users = DataIterator(list(self.user_pos_test.keys()), batch_size=2048, shuffle=False, drop_last=False)
        batch_result = []
        for batch_users in test_users:
            if self.user_neg_test is not None:
                candidate_items = []
                for user in batch_users:
                    num_item = len(self.user_pos_test[user])
                    if num_item != 1:
                        raise ValueError("the number of test item of user %d is %d" % (user, num_item))
                    candidate_items.append([self.user_pos_test[user][0]] + self.user_neg_test[user])
                test_items = [0] * len(batch_users)
                ranking_score = model.predict(batch_users, candidate_items)  # (B,N)
                ranking_score = np.array(ranking_score)
            else:
                test_items = []
                for user in batch_users:
                    num_item = len(self.user_pos_test[user])
                    if num_item != 1:
                        raise ValueError("the number of test item of user %d is %d" % (user, num_item))
                    test_items.append(self.user_pos_test[user][0])
                ranking_score = model.predict(batch_users, None)  # (B,N)
                ranking_score = np.array(ranking_score)

                # set the ranking scores of training items to -inf,
                # then the training items will be sorted at the end of the ranking list.
                for idx, user in enumerate(batch_users):
                    train_items = self.user_pos_train[user]
                    ranking_score[idx][train_items] = -np.inf

            result = eval_score_matrix_loo(ranking_score, test_items, top_k=self.max_top, thread_num=None)  # (B,k*metric_num)
            batch_result.append(result)

        # concatenate the batch results to a matrix
        all_user_result = np.concatenate(batch_result, axis=0)
        final_result = np.mean(all_user_result, axis=0)  # mean

        final_result = np.reshape(final_result, newshape=[self.metrics_num, self.max_top])
        final_result = final_result[:, self.top_show-1]
        final_result = np.reshape(final_result, newshape=[-1])
        buf = '\t'.join(["%.8f" % x for x in final_result])
        return buf
