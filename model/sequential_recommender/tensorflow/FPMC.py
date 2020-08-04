"""
Paper: Factorizing Personalized Markov Chains for Next-Basket Recommendation
Author: Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["FPMC"]


import tensorflow as tf
from model.base import AbstractRecommender
from util.tensorflow import l2_loss, inner_product
from util.tensorflow import pairwise_loss, pointwise_loss
from util.tensorflow import get_initializer, get_session
from util.common import Reduction
from data import TimeOrderPairwiseSampler, TimeOrderPointwiseSampler


class FPMC(AbstractRecommender):
    def __init__(self, config):
        super(FPMC, self).__init__(config)
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.emb_size = config["embedding_size"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        self.user_pos_dict = self.dataset.train_data.to_user_dict(by_time=True)

        self._build_model()
        self.sess = get_session(config["gpu_mem"])

    def _create_variable(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.last_item_ph = tf.placeholder(tf.int32, [None], name="item_last")  # the previous item
        self.pos_item_ph = tf.placeholder(tf.int32, [None], name="item_pos")  # the next item
        self.neg_item_ph = tf.placeholder(tf.int32, [None], name="item_neg")  # the negative item
        self.labels_ph = tf.placeholder(tf.int32, [None], name="label")  # the label

        init = get_initializer(self.param_init)
        self.UI_embeddings = tf.Variable(init([self.num_users, self.emb_size]), name="UI_embeddings")
        self.IU_embeddings = tf.Variable(init([self.num_items, self.emb_size]), name="IU_embeddings")
        self.IL_embeddings = tf.Variable(init([self.num_items, self.emb_size]), name="IL_embeddings")
        self.LI_embeddings = tf.Variable(init([self.num_items, self.emb_size]), name="LI_embeddings")

    def _build_model(self):
        self._create_variable()
        ui_emb = tf.nn.embedding_lookup(self.UI_embeddings, self.user_ph)  # b*d
        pos_iu_emb = tf.nn.embedding_lookup(self.IU_embeddings, self.pos_item_ph)  # b*d
        neg_iu_emb = tf.nn.embedding_lookup(self.IU_embeddings, self.neg_item_ph)  # b*d
        pos_il_emb = tf.nn.embedding_lookup(self.IL_embeddings, self.pos_item_ph)  # b*d
        neg_il_emb = tf.nn.embedding_lookup(self.IL_embeddings, self.neg_item_ph)  # b*d
        last_emb = tf.nn.embedding_lookup(self.LI_embeddings, self.last_item_ph)  # b*d

        y_pos = inner_product(ui_emb, pos_iu_emb) + inner_product(last_emb, pos_il_emb)
        y_neg = inner_product(ui_emb, neg_iu_emb) + inner_product(last_emb, neg_il_emb)

        reg_loss = l2_loss(ui_emb, pos_iu_emb, pos_il_emb, last_emb)

        if self.is_pairwise:
            model_loss = pairwise_loss(self.loss_func, y_pos-y_neg, reduction=Reduction.SUM)
            reg_loss += l2_loss(neg_iu_emb, neg_il_emb)
        else:
            model_loss = pointwise_loss(self.loss_func, y_pos, self.labels_ph, reduction=Reduction.SUM)

        final_loss = model_loss + self.reg*reg_loss

        self.train_opt = tf.train.AdamOptimizer(self.lr).minimize(final_loss, name="train_opt")

        # for evaluation
        self.batch_ratings = tf.matmul(ui_emb, self.IU_embeddings, transpose_b=True) + \
                             tf.matmul(last_emb, self.IL_embeddings, transpose_b=True)

    def train_model(self):
        if self.is_pairwise:
            self._train_pairwise()
        else:
            self._train_pointwise()

    def _train_pairwise(self):
        data_iter = TimeOrderPairwiseSampler(self.dataset.train_data,
                                             len_seqs=1, len_next=1,  num_neg=1,
                                             batch_size=self.batch_size,
                                             shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            for bat_users, bat_last_items, bat_pos_items, bat_neg_items in data_iter:
                feed = {self.user_ph: bat_users,
                        self.last_item_ph: bat_last_items,
                        self.pos_item_ph: bat_pos_items,
                        self.neg_item_ph: bat_neg_items}
                self.sess.run(self.train_opt, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _train_pointwise(self):
        data_iter = TimeOrderPointwiseSampler(self.dataset.train_data,
                                              len_seqs=1, len_next=1, num_neg=1,
                                              batch_size=self.batch_size,
                                              shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            for bat_users, bat_last_items, bat_items, bat_labels in data_iter:
                feed = {self.user_ph: bat_users,
                        self.last_item_ph: bat_last_items,
                        self.pos_item_ph: bat_items,
                        self.labels_ph: bat_labels}
                self.sess.run(self.train_opt, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users):
        last_items = [self.user_pos_dict[u][-1] for u in users]
        feed_dict = {self.user_ph: users, self.last_item_ph: last_items}
        all_ratings = self.sess.run(self.batch_ratings, feed_dict=feed_dict)
        return all_ratings
