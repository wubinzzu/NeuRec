from evaluator import ProxyEvaluator
import pandas as pd
import numpy as np
import scipy.sparse as sp
from util import Logger
import os
import time

def _create_logger(config, data_name):
    # create a logger
    timestamp = time.time()
    param_str = "%s_%s" % (data_name, config.params_str())
    run_id = "%s_%.8f" % (param_str[:150], timestamp)

    model_name = config["recommender"]
    log_dir = os.path.join("log", data_name, model_name)
    logger_name = os.path.join(log_dir, run_id + ".log")
    logger = Logger(logger_name)

    return logger


class AbstractRecommender(object):
    def __init__(self, dataset, conf):
        self.evaluator = ProxyEvaluator(dataset.get_user_train_dict(),
                                        dataset.get_user_test_dict(),
                                        dataset.get_user_test_neg_dict(),
                                        metric=conf["metric"],
                                        group_view=conf["group_view"],
                                        top_k=conf["topk"],
                                        batch_size=conf["test_batch_size"],
                                        num_thread=conf["num_thread"])

        self.logger = _create_logger(conf, dataset.dataset_name)
        self.logger.info(dataset)
        self.logger.info(conf)

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
