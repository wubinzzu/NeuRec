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
from neurec.util import learner,data_gen
from neurec.evaluation import Evaluate
from neurec.model.AbstractRecommender import AbstractRecommender
from neurec.util.properties import Properties
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class APR(AbstractRecommender):
    properties = [
        "learning_rate",
        "embedding_size",
        "learner",
        "topk",
        "epochs",
        "eps",
        "adv",
        "adver",
        "adv_epoch",
        "reg",
        "reg_adv",
        "batch_size",
        "verbose",
        "loss_function"
    ]

    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.learning_rate = self.conf["learning_rate"]
        self.embedding_size = self.conf["embedding_size"]
        self.learner = self.conf["learner"]
        self.topK = self.conf["topk"]
        self.num_epochs= self.conf["epochs"]
        self.eps= self.conf["eps"]
        self.adv = self.conf["adv"]
        self.adver = self.conf["adver"]
        self.adv_epoch = self.conf["adv_epoch"]
        self.reg = self.conf["reg"]
        self.reg_adv = self.conf["reg_adv"]
        self.batch_size= self.conf["batch_size"]
        self.verbose= self.conf["verbose"]
        self.loss_function = self.conf["loss_function"]
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None,], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None,], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None,], name="item_input_neg")

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

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            self.embedding_p = tf.nn.embedding_lookup(self.embedding_P, self.user_input)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, item_input)  # (b, embedding_size)
            return tf.reduce_sum(self.embedding_p * self.embedding_q,1) # (b, embedding_size) * (embedding_size, 1)

    def _create_inference_adv(self, item_input):
        with tf.name_scope("inference_adv"):
            # embedding look up
            self.embedding_p = tf.nn.embedding_lookup(self.embedding_P, self.user_input)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, item_input)  # (b, embedding_size)
            # add adversarial noise
            self.P_plus_delta = self.embedding_p + tf.nn.embedding_lookup(self.delta_P, self.user_input)
            self.Q_plus_delta = self.embedding_q + tf.nn.embedding_lookup(self.delta_Q, item_input)
            return tf.reduce_sum(self.P_plus_delta * self.Q_plus_delta,1) # (b, embedding_size) * (embedding_size, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            self.output = self._create_inference(self.item_input_pos)
            self.output_neg = self._create_inference(self.item_input_neg)
            self.result = self.output - self.output_neg
            # self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) # this is numerically unstable
            self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

            # loss to be omptimized
            self.opt_loss = self.loss + self.reg * (
                        tf.reduce_sum(tf.square(self.embedding_P)) + tf.reduce_sum(tf.square(self.embedding_Q)))

            if self.adver:
                # loss for L(Theta + adv_Delta)
                self.output_adv = self._create_inference_adv(self.item_input_pos)
                self.output_neg_adv = self._create_inference_adv(self.item_input_neg)
                self.result_adv = self.output_adv - self.output_neg_adv
                # self.loss_adv = tf.reduce_sum(tf.log(1 + tf.exp(-self.result_adv)))
                self.loss_adv = tf.reduce_sum(tf.nn.softplus(-self.result_adv))
                self.opt_loss += self.reg_adv * self.loss_adv


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
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self._create_adversarial()
#---------- training process -------
    def train_model(self):
        for epoch in  range(self.num_epochs):
            # Generate training instances
            user_input, item_input_pos, item_input_neg = data_gen._get_pairwise_all_data(self.dataset)
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                bat_users,bat_items_pos,bat_items_neg =\
                 data_gen._get_pairwise_batch_data(user_input,\
                 item_input_pos, item_input_neg, num_batch, self.batch_size)
                feed_dict = {self.user_input:bat_users,self.item_input_pos:bat_items_pos,\
                            self.item_input_neg:bat_items_neg}

                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
            self.logger.info("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)

    def predict(self, user_id, items):
        users = np.full(len(items), user_id, dtype=np.int32)
        return self.sess.run(self.output, feed_dict={self.user_input: users, self.item_input_pos: items})
