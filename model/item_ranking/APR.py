'''
Reference: Xiangnan He, et al., Adversarial Personalized Ranking for Recommendation" in SIGIR2018
@author: wubin
'''
from __future__ import absolute_import
from __future__ import division
import os
import tensorflow as tf
import numpy as np
from time import time
import configparser
from util import learner,data_gen
from evaluation import Evaluate
from model.AbstractRecommender import AbstractRecommender
from util.tool import random_choice, csr_to_user_dict
from util.dataiterator import DataIterator
from util.Logger import logger
from concurrent.futures import ThreadPoolExecutor
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class APR(AbstractRecommender):
    def __init__(self,sess,dataset):
        config = configparser.ConfigParser()
        config.read("conf/APR.properties")
        self.conf=dict(config.items("hyperparameters"))
        # print("APR arguments: %s " %(self.conf))
        self.learning_rate = float(self.conf["learning_rate"])
        self.embedding_size = int(self.conf["embedding_size"])
        self.learner = self.conf["learner"]
        self.topK = int(self.conf["topk"])
        self.num_epochs = int(self.conf["epochs"])
        self.eps = float(self.conf["eps"])
        self.adv = str(self.conf["adv"])
        self.dns = int(self.conf["dns"])
        self.adver = int(self.conf["adver"])
        self.adv_epoch = int(self.conf["adv_epoch"])
        self.reg = float(self.conf["reg"])
        self.reg_adv = float(self.conf["reg_adv"])
        self.batch_size = int(self.conf["batch_size"])
        self.verbose = int(self.conf["verbose"])
        self.loss_function = self.conf["loss_function"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.sess = sess
        self.user_pos_train = csr_to_user_dict(dataset.trainMatrix.tocsr())
        self.all_items = np.arange(self.num_items)

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32)  # (items, embedding_size)

            self.delta_P = tf.Variable(tf.zeros(shape=[self.num_users, self.embedding_size]),
                                       name='delta_P', dtype=tf.float32, trainable=False)  # (users, embedding_size)
            self.delta_Q = tf.Variable(tf.zeros(shape=[self.num_items, self.embedding_size]),
                                       name='delta_Q', dtype=tf.float32, trainable=False)  # (items, embedding_size)

            self.h = tf.constant(1.0, tf.float32, [self.embedding_size, 1], name="h")

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)  # (b, embedding_size)
            return tf.matmul(self.embedding_p * self.embedding_q, self.h), self.embedding_p, self.embedding_q  # (b, embedding_size) * (embedding_size, 1)

    def _create_inference_adv(self, item_input):
        with tf.name_scope("inference_adv"):
            # embedding look up
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)  # (b, embedding_size)
            # add adversarial noise
            self.P_plus_delta = self.embedding_p + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_P, self.user_input),
                                                                 1)
            self.Q_plus_delta = self.embedding_q + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_Q, item_input), 1)
            return tf.matmul(self.P_plus_delta * self.Q_plus_delta, self.h), self.embedding_p, self.embedding_q  # (b, embedding_size) * (embedding_size, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            self.output, embed_p_pos, embed_q_pos = self._create_inference(self.item_input_pos)
            self.output_neg, embed_p_neg, embed_q_neg = self._create_inference(self.item_input_neg)
            self.result = tf.clip_by_value(self.output - self.output_neg, -80.0, 1e8)
            # self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) # this is numerically unstable
            self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

            # loss to be omptimized
            self.pretrain_loss = self.loss + self.reg * tf.reduce_mean(tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg)) # embed_p_pos == embed_q_neg

            # loss for L(Theta + adv_Delta)
            self.output_adv, embed_p_pos, embed_q_pos = self._create_inference_adv(self.item_input_pos)
            self.output_neg_adv, embed_p_neg, embed_q_neg = self._create_inference_adv(self.item_input_neg)
            self.result_adv = tf.clip_by_value(self.output_adv - self.output_neg_adv, -80.0, 1e8)
            # self.loss_adv = tf.reduce_sum(tf.log(1 + tf.exp(-self.result_adv)))
            self.loss_adv = tf.reduce_sum(tf.nn.softplus(-self.result_adv))
            self.amf_loss = self.pretrain_loss + self.reg_adv * self.loss_adv + self.reg * \
                            tf.reduce_mean(tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

    def _create_adversarial(self):
        with tf.name_scope("adversarial"):
            # generate the adversarial weights by random method
            if self.adv == "random":
                # generation
                self.adv_P = tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01)
                self.adv_Q = tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01)

                # normalization and multiply epsilon
                self.update_P = self.delta_P.assign(tf.nn.l2_normalize(self.adv_P, 1) * self.eps)
                self.update_Q = self.delta_Q.assign(tf.nn.l2_normalize(self.adv_Q, 1) * self.eps)

            # generate the adversarial weights by gradient-based method
            elif self.adv == "grad":
                # return the IndexedSlice Data: [(values, indices, dense_shape)]
                # grad_var_P: [grad,var], grad_var_Q: [grad, var]
                self.grad_P, self.grad_Q = tf.gradients(self.loss, [self.embedding_P, self.embedding_Q])

                # convert the IndexedSlice Data to Dense Tensor
                self.grad_P_dense = tf.stop_gradient(self.grad_P)
                self.grad_Q_dense = tf.stop_gradient(self.grad_Q)

                # normalization: new_grad = (grad / |grad|) * eps
                self.update_P = self.delta_P.assign(tf.nn.l2_normalize(self.grad_P_dense, 1) * self.eps)
                self.update_Q = self.delta_Q.assign(tf.nn.l2_normalize(self.grad_Q_dense, 1) * self.eps)

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.bpr_optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.pretrain_loss)
            self.amf_optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.amf_loss)
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self._create_adversarial()

    def train_model(self):
        self._pre_training()
        self._adversarial_training()

    def _pre_training(self):
        # pretrain
        logger.info("Pre-training")
        for epoch in range(self.adv_epoch):
            data = self.get_train_data()
            for user_input, item_input_pos, item_dns_list in data:
                user_feed = user_input
                item_pos_feed = item_input_pos
                if self.dns == 1:
                    item_neg_feed = [neg[0] for neg in item_dns_list]
                elif self.dns > 1:
                    user_tmp = []
                    neg_tmp = []
                    for user, neg in zip(user_input, item_dns_list):
                        user_tmp.extend([user] * self.dns)
                        neg_tmp.extend(neg)

                    user_tmp = np.reshape(user_tmp, newshape=[-1, 1])
                    neg_tmp = np.reshape(neg_tmp, newshape=[-1, 1])
                    feed_dict = {self.user_input: user_tmp,
                                 self.item_input_neg: neg_tmp}
                    output_neg = self.sess.run(self.output_neg, feed_dict)
                    # select the best negtive sample as for item_input_neg
                    output_neg = np.reshape(output_neg, newshape=[-1, self.dns])
                    max_idx = np.argmax(output_neg, axis=1)

                    item_neg_feed = [neg_list[idx] for idx, neg_list in zip(max_idx, item_dns_list)]

                user_feed = np.reshape(user_feed, newshape=[-1, 1])
                item_pos_feed = np.reshape(item_pos_feed, newshape=[-1, 1])
                item_neg_feed = np.reshape(item_neg_feed, newshape=[-1, 1])
                feed_dict = {self.user_input: user_feed,
                             self.item_input_pos: item_pos_feed,
                             self.item_input_neg: item_neg_feed}
                self.sess.run(self.bpr_optimizer, feed_dict)

            if epoch % self.verbose == 0:
                Evaluate.test_model(self,self.dataset)

    def _adversarial_training(self):
        # adversarial training
        logger.info("Adversarial training")
        for epoch in range(self.adv_epoch, self.num_epochs):
            data = self.get_train_data()
            for user_input, item_input_pos, item_dns_list in data:
                user_feed = user_input
                item_pos_feed = item_input_pos
                item_neg_feed = [neg[0] for neg in item_dns_list]

                user_feed = np.reshape(user_feed, newshape=[-1, 1])
                item_pos_feed = np.reshape(item_pos_feed, newshape=[-1, 1])
                item_neg_feed = np.reshape(item_neg_feed, newshape=[-1, 1])

                feed_dict = {self.user_input: user_feed,
                             self.item_input_pos: item_pos_feed,
                             self.item_input_neg: item_neg_feed}

                self.sess.run([self.update_P, self.update_Q], feed_dict)
                self.sess.run(self.amf_optimizer, feed_dict)

            if epoch % self.verbose == 0:
                Evaluate.test_model(self,self.dataset)

    def get_train_data(self):
        users_list, pos_items, neg_items = [], [], []
        train_users = list(self.user_pos_train.keys())
        with ThreadPoolExecutor() as executor:
            data = executor.map(self.get_train_data_one_user, train_users)
        data = list(data)
        for users, pos, neg_dns in data:
            users_list.extend(users)
            pos_items.extend(pos)
            neg_items.extend(neg_dns)

        dataloader = DataIterator(users_list, pos_items, neg_items, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def get_train_data_one_user(self, user):
        pos = self.user_pos_train[user]
        pos_len = len(pos)

        neg = random_choice(self.all_items, size=pos_len*self.dns, exclusion=pos)
        neg = np.reshape(neg, newshape=[pos_len, self.dns])
        return [user] * pos_len, pos.tolist(), neg.tolist()

    def predict(self, user_id, items):
        items = np.reshape(items, newshape=[-1, 1])
        users = np.full(np.shape(items), user_id, dtype=np.int32)
        return self.sess.run(self.output, feed_dict={self.user_input: users, self.item_input_pos: items})  