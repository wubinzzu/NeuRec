"""
Paper: FISM: Factored Item Similarity Models for Top-N Recommender Systems
Author: Santosh Kabbur, Xia Ning, and George Karypis
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["FISM"]

import tensorflow as tf
from model.base import AbstractRecommender
from util.tensorflow import inner_product, l2_loss
from util.tensorflow import pairwise_loss, pointwise_loss
from util.tensorflow import get_initializer, get_session
from util.common import Reduction
from data import FISMPairwiseSampler, FISMPointwiseSampler
from reckit import pad_sequences
import numpy as np


class FISM(AbstractRecommender):
    def __init__(self, config):
        super(FISM, self).__init__(config)
        self.emb_size = config["embedding_size"]
        self.lr = config["lr"]
        self.alpha = config["alpha"]
        self.reg_lambda = config["reg_lambda"]
        self.reg_gamma = config["reg_gamma"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.pad_idx = self.num_items
        self.user_train_dict = self.dataset.train_data.to_user_dict()

        self._build_model()
        self.sess = get_session(config["gpu_mem"])

    def _create_variable(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user_ph")
        self.his_items_ph = tf.placeholder(tf.int32, [None, None], name="historical_items")
        self.his_len_ph = tf.placeholder(tf.float32, [None], name="len_of_his_items")
        self.pos_item_ph = tf.placeholder(tf.int32, [None], name="pos_item")
        self.neg_item_ph = tf.placeholder(tf.int32, [None], name="neg_item")
        self.label_ph = tf.placeholder(tf.float32, [None], name="label")

        # embedding layers
        init = get_initializer(self.param_init)
        zero_init = get_initializer("zeros")
        his_embeddings = tf.Variable(init([self.num_users, self.emb_size]), name="his_embedding")
        zero_pad = tf.Variable(zero_init([1, self.emb_size]), trainable=False, name="zero_pad")
        self.his_embeddings = tf.concat([his_embeddings, zero_pad], axis=0)

        self.item_embeddings = tf.Variable(init([self.num_items, self.emb_size]), name="item_embedding")
        self.user_biases = tf.Variable(zero_init([self.num_users]), name="user_bias")
        self.item_biases = tf.Variable(zero_init([self.num_items]), name="item_bias")

    def _build_model(self):
        self._create_variable()
        his_embs = tf.nn.embedding_lookup(self.his_embeddings, self.his_items_ph)  # (b,l,d)
        user_emb = tf.reduce_sum(his_embs, axis=1)  # (b,d)
        scale = tf.reshape(tf.pow(self.his_len_ph, -self.alpha), shape=[-1, 1])  # (b,1)
        user_emb = tf.multiply(user_emb, scale)
        user_bias = tf.gather(self.user_biases, self.user_ph)  # (b,)

        pos_item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.pos_item_ph)  # (b,d)
        pos_bias = tf.gather(self.item_biases, self.pos_item_ph)  # (b,)
        neg_item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.neg_item_ph)  # (b,d)
        neg_bias = tf.gather(self.item_biases, self.neg_item_ph)  # (b,)

        hat_yi = inner_product(user_emb, pos_item_emb) + pos_bias + user_bias
        hat_yj = inner_product(user_emb, neg_item_emb) + neg_bias + user_bias

        # reg loss
        emb_reg = l2_loss(his_embs, pos_item_emb)
        bias_reg = l2_loss(pos_bias)
        if self.is_pairwise:
            model_loss = pairwise_loss(self.loss_func, hat_yi-hat_yj, reduction=Reduction.SUM)
            emb_reg += l2_loss(neg_item_emb)
            bias_reg += l2_loss(neg_bias)
        else:
            model_loss = pointwise_loss(self.loss_func, hat_yi, self.label_ph, reduction=Reduction.SUM)
            bias_reg += l2_loss(user_bias)

        final_loss = model_loss + self.reg_lambda*emb_reg + self.reg_gamma*bias_reg

        self.train_opt = tf.train.AdamOptimizer(self.lr).minimize(final_loss, name="train_opt")

        # for evaluation
        self.batch_ratings = tf.matmul(user_emb, self.item_embeddings, transpose_b=True) + self.item_biases

    def train_model(self):
        if self.is_pairwise:
            self._train_pairwise()
        else:
            self._train_pointwise()

    def _train_pairwise(self):
        data_iter = FISMPairwiseSampler(self.dataset.train_data, pad=self.pad_idx,
                                        batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            for bat_users, bat_his_items, bat_his_len, bat_pos_items, bat_neg_items in data_iter:
                feed = {self.user_ph: bat_users,
                        self.his_items_ph: bat_his_items,
                        self.his_len_ph: bat_his_len,
                        self.pos_item_ph: bat_pos_items,
                        self.neg_item_ph: bat_neg_items
                        }
                self.sess.run(self.train_opt, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _train_pointwise(self):
        data_iter = FISMPointwiseSampler(self.dataset.train_data, pad=self.pad_idx,
                                         batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            for bat_users, bat_his_items, bat_his_len, bat_items, bat_labels in data_iter:
                feed = {self.user_ph: bat_users,
                        self.his_items_ph: bat_his_items,
                        self.his_len_ph: bat_his_len,
                        self.pos_item_ph: bat_items,
                        self.label_ph: bat_labels
                        }
                self.sess.run(self.train_opt, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users):
        his_items = [self.user_train_dict[u] for u in users]
        his_len = [len(self.user_train_dict[u]) for u in users]
        his_items = pad_sequences(his_items, value=self.pad_idx, max_len=None,
                                  padding='post', truncating='post', dtype=np.int32)
        feed = {self.user_ph: users,
                self.his_items_ph: his_items,
                self.his_len_ph: his_len}
        all_ratings = self.sess.run(self.batch_ratings, feed_dict=feed)
        return all_ratings
