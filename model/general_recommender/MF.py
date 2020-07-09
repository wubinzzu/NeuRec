"""
Reference: Steffen Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
    GMF: Xiangnan He et al., "Neural Collaborative Filtering." in WWW 2017.
@author: wubin
"""
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool
from model.AbstractRecommender import AbstractRecommender
from util import timer
from util import l2_loss
from data import PairwiseSampler, PointwiseSampler


class MF(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(MF, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.embedding_size = conf["embedding_size"]
        self.learner = conf["learner"]
        self.loss_function = conf["loss_function"]
        self.is_pairwise = conf["is_pairwise"]
        self.num_epochs = conf["epochs"]
        self.reg_mf = conf["reg_mf"]
        self.batch_size = conf["batch_size"]
        self.verbose = conf["verbose"]
        self.num_negatives = conf["num_negatives"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.sess = sess
    
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None], name="item_input")
            if self.is_pairwise is True:
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")
            else:
                self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            initializer = tool.get_initializer(self.init_method, self.stddev)
            
            self.user_embeddings = tf.Variable(initializer([self.num_users, self.embedding_size]), 
                                               name='user_embeddings', dtype=tf.float32)  # (users, embedding_size)
            self.item_embeddings = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                               name='item_embeddings', dtype=tf.float32)  # (items, embedding_size)

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, item_input)
            predict = tf.reduce_sum(tf.multiply(user_embedding, item_embedding), 1)
            return user_embedding, item_embedding, predict

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            p1, q1, self.output = self._create_inference(self.item_input)
            if self.is_pairwise is True:
                _, q2, self.output_neg = self._create_inference(self.item_input_neg)
                result = self.output - self.output_neg
                self.loss = learner.pairwise_loss(self.loss_function, result) + self.reg_mf * l2_loss(p1, q2, q1)
            else:
                self.loss = learner.pointwise_loss(self.loss_function, self.labels, self.output) + \
                            self.reg_mf * l2_loss(p1, q1)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        
    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        if self.is_pairwise is True:
            data_iter = PairwiseSampler(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True)
        else:
            data_iter = PointwiseSampler(self.dataset, neg_num=self.num_negatives, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.num_epochs+1):
            total_loss = 0.0
            training_start_time = time()

            if self.is_pairwise is True:
                for bat_users, bat_items_pos, bat_items_neg in data_iter:
                    feed_dict = {self.user_input: bat_users,
                                 self.item_input: bat_items_pos,
                                 self.item_input_neg: bat_items_neg}
                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
            else:
                for bat_users, bat_items, bat_labels in data_iter:
                    feed_dict = {self.user_input: bat_users,
                                 self.item_input: bat_items,
                                 self.labels: bat_labels}
                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/len(data_iter),
                                                             time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
    
    @timer
    def evaluate(self):
        self._cur_user_embeddings, self._cur_item_embeddings = self.sess.run([self.user_embeddings, self.item_embeddings])
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items=None):
        user_embed = self._cur_user_embeddings[user_ids]
        ratings = np.matmul(user_embed, self._cur_item_embeddings.T)
        if candidate_items is not None:
            ratings = [rating[items] for rating, items in zip(ratings, candidate_items)]
        #     for idx, items in enumerate():
        #         ratings[idx]
        # else:
        #     ratings = []
        #     for user_id, items_by_user_id in zip(user_ids, candidate_items):
        #         user_embed = self._cur_user_embeddings[user_id]
        #         items_embed = self._cur_item_embeddings[items_by_user_id]
        #         ratings.append(np.squeeze(np.matmul(user_embed, items_embed.T)))

        return ratings
