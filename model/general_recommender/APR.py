"""
Reference: Xiangnan He, et al., Adversarial Personalized Ranking for Recommendation" in SIGIR2018
@author: wubin
"""
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool
from model.AbstractRecommender import AbstractRecommender
from util import timer
from util import l2_loss
from data import PairwiseSampler


class APR(AbstractRecommender):
    def __init__(self, sess, dataset, conf):  
        super(APR, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.embedding_size = conf["embedding_size"]
        self.learner = conf["learner"]
        self.num_epochs = conf["epochs"]
        self.eps = conf["eps"]
        self.adv = conf["adv"]
        self.adver = conf["adver"]
        self.adv_epoch = conf["adv_epoch"]
        self.reg = conf["reg"]
        self.reg_adv = conf["reg_adv"]
        self.batch_size = conf["batch_size"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.verbose = conf["verbose"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.sess = sess
    
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None,], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None,], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None,], name="item_input_neg")
            
    def _create_variables(self):
        with tf.name_scope("embedding"):
            initializer = tool.get_initializer(self.init_method, self.stddev)
            self.embedding_P = tf.Variable(initializer([self.num_users, self.embedding_size]),
                                           name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
            self.embedding_Q = tf.Variable(initializer([self.num_items, self.embedding_size]),
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
            self.opt_loss = self.loss + self.reg * l2_loss(self.embedding_P, self.embedding_Q)

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

    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        data_iter = PairwiseSampler(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.num_epochs+1):
            # Generate training instances
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(data_iter)
            for bat_users, bat_items_pos, bat_items_neg in data_iter:
                    feed_dict = {self.user_input: bat_users,
                                 self.item_input_pos: bat_items_pos,
                                 self.item_input_neg: bat_items_neg}
                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss

            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/num_training_instances,
                                                             time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
                
    @timer
    def evaluate(self):
        self._cur_user_embeddings, self._cur_item_embeddings = self.sess.run([self.embedding_P, self.embedding_Q])
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_userids):
        if candidate_items_userids is None:
            user_embed = self._cur_user_embeddings[user_ids]
            ratings = np.matmul(user_embed, self._cur_item_embeddings.T)
        else:
            ratings = []
            for userid, items_by_userid in zip(user_ids, candidate_items_userids):
                user_embed = self._cur_user_embeddings[userid]
                items_embed = self._cur_item_embeddings[items_by_userid]
                ratings.append(np.squeeze(np.matmul(user_embed, items_embed.T)))
            
        return ratings
