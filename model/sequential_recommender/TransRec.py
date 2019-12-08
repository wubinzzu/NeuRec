"""
Reference: Ruining He et al., "Translation-based Recommendation." in RecSys 2017
@author: wubin
"""

from model.AbstractRecommender import SeqAbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import Learner, DataGenerator, Tool
from util.Logger import logger
from util.DataIterator import DataIterator
from util.Tool import csr_to_user_dict_bytime
from util import timer
from util import l2_loss


class TransRec(SeqAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(TransRec, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.embedding_size = conf["embedding_size"]
        self.learner = conf["learner"]
        self.loss_function = conf["loss_function"]
        self.is_pairwise = conf["is_pairwise"]
        self.num_epochs = conf["epochs"]
        self.reg_mf = conf["reg_mf"]
        self.batch_size = conf["batch_size"]
        self.verbose = conf["verbose"]
        self.num_negatives = conf["num_neg"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.dataset = dataset 
        self.train_matrix = dataset.train_matrix
        self.train_dict = csr_to_user_dict_bytime(dataset.time_matrix, dataset.train_matrix)
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None], name="item_input_pos")
            self.item_input_recent = tf.placeholder(tf.int32, shape=[None], name="item_input_recent")
            if self.is_pairwise.lower() == "true":
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")
            else:
                self.labels = tf.placeholder(tf.float32, shape=[None, ], name="labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            initializer = Tool.get_initializer(self.init_method, self.stddev)
            self.user_embeddings = tf.Variable(initializer([self.num_users, self.embedding_size]),
                                               name='user_embeddings', dtype=tf.float32)  # (users, embedding_size)
            self.item_embeddings = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                               name='item_embeddings', dtype=tf.float32)  # (items, embedding_size)
            self.item_biases = tf.Variable(initializer([self.num_items]),
                                               name='item_biases', dtype=tf.float32)  # (items)
            self.global_embedding = tf.Variable(initializer([1, self.embedding_size]),
                                               name='global_embedding', dtype=tf.float32)
    
    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            batch_size = tf.shape(user_embedding)[0]
            item_embedding_recent = tf.nn.embedding_lookup(self.item_embeddings, self.item_input_recent)
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, item_input)
            
            item_bias = tf.nn.embedding_lookup(self.item_biases, item_input)
            predict_vector = user_embedding + tf.tile(self.global_embedding, tf.stack([batch_size, 1])) + item_embedding_recent - item_embedding
            predict = item_bias-tf.reduce_sum(tf.square(predict_vector), 1)
            return user_embedding, item_embedding_recent, item_embedding, item_bias, predict

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            p1, r1, q1, b1, self.output = self._create_inference(self.item_input)
            if self.is_pairwise.lower() == "true":
                _, _, q2, b2, output_neg = self._create_inference(self.item_input_neg)
                self.result = self.output - output_neg
                self.loss = Learner.pairwise_loss(self.loss_function,self.result) + \
                            self.reg_mf * l2_loss(p1, r1, q2, q1, b1, b2, self.global_embedding)
            else:
                self.loss = Learner.pointwise_loss(self.loss_function, self.labels, self.output) + \
                            self.reg_mf * l2_loss(p1, r1, q1, b1, self.global_embedding)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = Learner.optimizer(self.learner, self.loss, self.learning_rate)
                
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()

    # ---------- training process -------
    def train_model(self):
        logger.info(self.evaluator.metrics_info())
        for epoch in range(self.num_epochs):
            # Generate training instances
            if self.is_pairwise.lower() == "true":
                user_input, item_input_pos, item_input_recent, item_input_neg = \
                    DataGenerator._get_pairwise_all_firstorder_data(self.dataset, self.train_dict)
                data_iter = DataIterator(user_input, item_input_pos, item_input_recent, item_input_neg,
                                         batch_size=self.batch_size, shuffle=True)
            else:
                user_input, item_input, item_input_recent, labels = \
                    DataGenerator._get_pointwise_all_firstorder_data(self.dataset, self.num_negatives)
                data_iter = DataIterator(user_input, item_input, item_input_recent, labels,
                                         batch_size=self.batch_size, shuffle=True)
           
            num_training_instances = len(user_input)
            total_loss = 0.0
            training_start_time = time()

            if self.is_pairwise.lower() == "true":
                for bat_users, bat_items_pos, bat_items_recent, bat_items_neg in data_iter:
                    feed_dict = {self.user_input: bat_users,
                                 self.item_input: bat_items_pos,
                                 self.item_input_recent: bat_items_recent,
                                 self.item_input_neg: bat_items_neg}

                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
            else:
                for bat_users, bat_items, bat_items_recent, bat_labels in data_iter:
                    feed_dict = {self.user_input: bat_users,
                                 self.item_input: bat_items,
                                 self.item_input_recent: bat_items_recent,
                                 self.labels: bat_labels}

                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss

            logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss / num_training_instances,
                                                             time() - training_start_time))
            
            if epoch % self.verbose == 0:
                logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_userids):
        ratings = []
        if candidate_items_userids is None:
            allitems = np.arange(self.num_items)
            for userid in user_ids:
                cand_items = self.train_dict[userid]
                item_recent = np.full(self.num_items, cand_items[-1], dtype=np.int32)
    
                users = np.full(self.num_items, userid, dtype=np.int32)
                feed_dict = {self.user_input: users,
                             self.item_input_recent: item_recent,
                             self.item_input: allitems}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))   
                
        else :
            for userid, items_by_userid in zip(user_ids, candidate_items_userids):
                cand_items = self.train_dict[userid]
                item_recent = np.full(len(candidate_items_userids), cand_items[-1], dtype=np.int32)
    
                users = np.full(len(items_by_userid), userid, dtype=np.int32)
                feed_dict = {self.user_input: users,
                             self.item_input_recent: item_recent,
                             self.item_input: items_by_userid}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))

        return ratings
