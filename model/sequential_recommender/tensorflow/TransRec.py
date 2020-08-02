"""
Paper: Translation-based Recommendation
Author: Ruining He, Wang-Cheng Kang, and Julian McAuley
Reference: https://drive.google.com/file/d/0B9Ck8jw-TZUEVmdROWZKTy1fcEE/view?usp=sharing
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["TransRec"]


import tensorflow as tf
from model.base import AbstractRecommender
from util.tensorflow import l2_loss, l2_distance
from util.tensorflow import pairwise_loss, pointwise_loss
from util.tensorflow import get_initializer
from util.common import Reduction
from data import TimeOrderPairwiseSampler, TimeOrderPointwiseSampler


class TransRec(AbstractRecommender):
    def __init__(self, config):
        super(TransRec, self).__init__(config)
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

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = config["gpu_mem"]
        self.sess = tf.Session(config=tf_config)
        self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _create_variable(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.last_item_ph = tf.placeholder(tf.int32, [None], name="item_last")  # the previous item
        self.pos_item_ph = tf.placeholder(tf.int32, [None], name="item_pos")  # the next item
        self.neg_item_ph = tf.placeholder(tf.int32, [None], name="item_neg")  # the negative item
        self.labels_ph = tf.placeholder(tf.int32, [None], name="label")  # the label

        init = get_initializer(self.param_init)
        zero_init = get_initializer("zeros")
        self.user_embeddings = tf.Variable(zero_init([self.num_users, self.emb_size]), name="user_embeddings")

        self.global_transition = tf.Variable(init([1, self.emb_size]), name="global_transition")

        self.item_embeddings = tf.Variable(init([self.num_items, self.emb_size]), name="item_embeddings")

        self.item_biases = tf.Variable(zero_init([self.num_items]), name="item_biases")

    def _build_model(self):
        self._create_variable()
        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)
        last_item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.last_item_ph)
        pos_item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.pos_item_ph)
        neg_item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.neg_item_ph)

        pos_item_bias = tf.gather(self.item_biases, self.pos_item_ph)
        neg_item_bias = tf.gather(self.item_biases, self.neg_item_ph)

        transed_emb = user_emb + self.global_transition + last_item_emb
        y_pos = -l2_distance(transed_emb, pos_item_emb) + pos_item_bias
        y_neg = -l2_distance(transed_emb, neg_item_emb) + neg_item_bias

        reg_loss = l2_loss(user_emb, self.global_transition, last_item_emb, pos_item_emb, pos_item_bias)
        if self.is_pairwise:
            model_loss = pairwise_loss(self.loss_func, y_pos-y_neg, reduction=Reduction.SUM)
            reg_loss += l2_loss(neg_item_emb, neg_item_bias)
        else:
            model_loss = pointwise_loss(self.loss_func, y_pos, self.labels_ph, reduction=Reduction.SUM)

        final_loss = model_loss + self.reg*reg_loss

        self.train_opt = tf.train.AdamOptimizer(self.lr).minimize(final_loss, name="train_opt")

        # for evaluation
        transed_emb = tf.expand_dims(transed_emb, axis=1)  # (b,1,d)
        all_item_emb = tf.expand_dims(self.item_embeddings, axis=0)  # (1,n,d)
        self.batch_ratings = -l2_distance(transed_emb, all_item_emb) + self.item_biases  # (b,n)

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

    def predict(self, users, neg_items=None):
        last_items = [self.user_pos_dict[u][-1] for u in users]
        feed_dict = {self.user_ph: users, self.last_item_ph: last_items}
        all_ratings = self.sess.run(self.batch_ratings, feed_dict=feed_dict)
        return all_ratings
