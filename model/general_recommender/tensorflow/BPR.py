"""
Paper: BPR: Bayesian Personalized Ranking from Implicit Feedback
Author: Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["BPR"]

import tensorflow as tf
from model.base import AbstractRecommender
from util.tensorflow_util import log_loss, inner_product, l2_loss
from data import PairwiseSampler


class BPR(AbstractRecommender):
    def __init__(self, config):
        super(BPR, self).__init__(config)
        self.factors_num = config["factors_num"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = config["gpu_mem"]
        self.sess = tf.Session(config=tf_config)
        self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _create_variable(self):
        self.user_h = tf.placeholder(tf.int32, [None], name="user")
        self.pos_item_h = tf.placeholder(tf.int32, [None], name="pos_item")
        self.neg_item_h = tf.placeholder(tf.int32, [None], name="neg_item")

        # embedding layers
        initializer = tf.initializers.random_uniform(minval=-0.05, maxval=0.05)
        self.user_embeddings = tf.get_variable("user_embedding", shape=[self.num_users, self.factors_num],
                                               dtype=tf.float32, initializer=initializer, trainable=True)
        self.item_embeddings = tf.get_variable("item_embedding", shape=[self.num_items, self.factors_num],
                                               dtype=tf.float32, initializer=initializer, trainable=True)
        self.item_biases = tf.get_variable("item_bias", shape=[self.num_items], dtype=tf.float32,
                                           initializer=tf.initializers.zeros(), trainable=True)

    def _build_model(self):
        self._create_variable()
        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_h)
        pos_item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.pos_item_h)
        neg_item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.neg_item_h)
        pos_bias = tf.gather(self.item_biases, self.pos_item_h)
        neg_bias = tf.gather(self.item_biases, self.neg_item_h)

        yi_hat = inner_product(user_emb, pos_item_emb) + pos_bias
        yj_hat = inner_product(user_emb, neg_item_emb) + neg_bias
        model_loss = tf.reduce_sum(log_loss(yi_hat - yj_hat))

        # reg loss
        unique_user, _ = tf.unique(tf.reshape(self.user_h, shape=[-1]))
        u_params = tf.nn.embedding_lookup(self.user_embeddings, unique_user)
        unique_item, _ = tf.unique(tf.reshape(tf.concat([self.pos_item_h, self.neg_item_h], axis=-1), shape=[-1]))
        i_params = tf.nn.embedding_lookup(self.item_embeddings, unique_item)

        reg_loss = l2_loss(u_params, i_params)

        self.final_loss = model_loss + self.reg * reg_loss

        self.update = tf.train.AdamOptimizer(self.lr).minimize(self.final_loss, name="update")

        # for evaluation
        u_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_h)
        self.all_logits_pre = tf.matmul(u_emb, self.item_embeddings, transpose_b=True) + self.item_biases

    def train_model(self):
        data_iter = PairwiseSampler(self.dataset.train_data, num_neg=1,
                                    batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                feed = {self.user_h: bat_users,
                        self.pos_item_h: bat_pos_items,
                        self.neg_item_h: bat_neg_items}
                self.sess.run(self.update, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, neg_items=None):
        all_ratings = self.sess.run(self.all_logits_pre, feed_dict={self.user_h: users})
        return all_ratings
