import numpy as np
from model.AbstractRecommender import AbstractRecommender


class Pop(AbstractRecommender):
    def __init__(self, sess, dataset, config):
        super(Pop, self).__init__(dataset, config)
        self.dataset = dataset
        self.users_num, self.items_num = self.dataset.num_users, self.dataset.num_items

    def build_graph(self):
        pass

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        item_csr = self.dataset.to_csr_matrix().transpose(copy=False)
        items_count = [item_csr[i].getnnz() for i in range(item_csr.shape[0])]
        self.ranking_score = np.array(items_count, dtype=np.float32)
        result = self.evaluate_model()
        self.logger.info("result:\t%s" % result)

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, items=None):
        ratings = np.tile(self.ranking_score, len(users))
        ratings = np.reshape(ratings, newshape=[len(users), self.items_num])

        if items is not None:
            ratings = [ratings[idx][u_item] for idx, u_item in enumerate(items)]
        return ratings
