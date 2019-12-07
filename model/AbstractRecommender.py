from evaluator import FoldOutEvaluator, LeaveOneOutEvaluator, SparsityEvaluator
import pandas as pd
import numpy as np
import scipy.sparse as sp


class AbstractRecommender(object):
    def __init__(self, dataset, conf):
        if conf["test_view"] is not None:
            self.evaluator = SparsityEvaluator(dataset.train_matrix, dataset.test_matrix, dataset.negative_matrix, conf)
        elif conf["splitter"] == "ratio":
            self.evaluator = FoldOutEvaluator(dataset.train_matrix, dataset.test_matrix, dataset.negative_matrix, conf)
        elif conf["splitter"] == "loo":
            self.evaluator = LeaveOneOutEvaluator(dataset.train_matrix, dataset.test_matrix, dataset.negative_matrix, conf)
        else:
            raise ValueError("There is not evaluator named '%s'" % conf["splitter"])

    def build_graph(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError
    
    def predict(self, user_ids, items):
        raise NotImplementedError


class SeqAbstractRecommender(AbstractRecommender):
    def __init__(self, dataset, conf):
        if dataset.time_matrix is None:
            raise ValueError("Dataset does not contant time infomation!")
        super(SeqAbstractRecommender, self).__init__(dataset, conf)


class SocialAbstractRecommender(AbstractRecommender):
    def __init__(self, dataset, conf):
        super(SocialAbstractRecommender, self).__init__(dataset, conf)
        social_users = pd.read_csv(conf["social_file"], sep=conf["data.convert.separator"],
                                   header=None, names=["user", "friend"])
        users_key = np.array(list(dataset.userids.keys()))
        index = np.in1d(social_users["user"], users_key)
        social_users = social_users[index]

        index = np.in1d(social_users["friend"], users_key)
        social_users = social_users[index]

        user = social_users["user"]
        user_id = [dataset.userids[u] for u in user]
        friend = social_users["friend"]
        friend_id = [dataset.userids[u] for u in friend]
        num_users, num_items = dataset.train_matrix.shape
        self.social_matrix = sp.csr_matrix(([1] * len(user_id), (user_id, friend_id)),
                                           shape=(num_users, num_users))
