"""
Reference: Ruining He et al., "Fusing similarity models with Markov chains for sparse sequential recommendation." in ICDM 2016.
@author: wubin
"""

from model.AbstractRecommender import SeqAbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool, data_generator
from util.tool import csr_to_user_dict_bytime, timer, pad_sequences
from util import l2_loss
from util.data_iterator import DataIterator


class Fossil(SeqAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(Fossil, self).__init__(dataset, conf)
        self.verbose = conf["verbose"]
        self.batch_size = conf["batch_size"]
        self.num_epochs = conf["epochs"]
        self.embedding_size = conf["embedding_size"]
        regs = conf["regs"]
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.reg_eta = regs[2]
        self.alpha = conf["alpha"]
        self.num_negatives = conf["num_neg"]
        self.learning_rate = conf["learning_rate"]
        self.learner = conf["learner"]
        self.loss_function = conf["loss_function"]
        self.is_pairwise = conf["is_pairwise"]
        self.high_order = conf["high_order"]
        self.num_negatives = conf["num_neg"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.num_negatives = conf["num_neg"]
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.dataset = dataset
        self.train_matrix = self.dataset.train_matrix
        self.train_dict = csr_to_user_dict_bytime(self.dataset.time_matrix,self.train_matrix)
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input_id = tf.placeholder(tf.int32, shape=[None, ], name="user_input_id")  # the index of users
            self.user_input = tf.placeholder(tf.int32, shape=[None, None], name="user_input")  # the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, ], name="num_idx")  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, ], name="item_input_pos")  # the index of items
            self.item_input_recent = tf.placeholder(tf.int32, shape=[None, None], name="item_input_recent")
            if self.is_pairwise is True:
                self.user_input_neg = tf.placeholder(tf.int32, shape=[None, None], name="user_input_neg")
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None, ], name="item_input_neg")
                self.num_idx_neg = tf.placeholder(tf.float32, shape=[None, ], name="num_idx_neg")
            else :
                self.labels = tf.placeholder(tf.float32, shape=[None, ], name="labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            initializer = tool.get_initializer(self.init_method, self.stddev)
            
            self.c1 = tf.Variable(initializer([self.num_items, self.embedding_size]), name='c1', dtype=tf.float32)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_P = tf.concat([self.c1, self.c2], 0, name='embedding_P')
            self.embedding_Q = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                           name='embedding_Q', dtype=tf.float32)
            
            self.eta = tf.Variable(initializer([self.num_users, self.high_order]), name='eta')
            self.eta_bias = tf.Variable(initializer([1, self.high_order]), name='eta_bias')
            
            self.bias = tf.Variable(tf.zeros(self.num_items), name='bias')

    def _create_inference(self, user_input, item_input, num_idx):
        with tf.name_scope("inference"):
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, user_input), 1)
            embedding_eta_u = tf.nn.embedding_lookup(self.eta, self.user_input_id)
            batch_size = tf.shape(embedding_eta_u)[0]
            eta = tf.expand_dims(tf.tile(self.eta_bias, tf.stack([batch_size, 1])) + embedding_eta_u, -1)
            embeddings_short = tf.nn.embedding_lookup(self.embedding_P, self.item_input_recent)
            embedding_q = tf.nn.embedding_lookup(self.embedding_Q, item_input)
            bias_i = tf.nn.embedding_lookup(self.bias, item_input)
            coeff = tf.pow(num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            output = coeff * tf.reduce_sum(tf.multiply(embedding_p, embedding_q), 1) +\
                     tf.reduce_sum(tf.multiply(tf.reduce_sum(eta*embeddings_short, 1), embedding_q), 1) + bias_i
        return embedding_p, embedding_q, embedding_eta_u, embeddings_short, output

    def _create_loss(self):
        with tf.name_scope("loss"):
            p1, q1, eta_u, short, self.output = self._create_inference(self.user_input, self.item_input, self.num_idx)
            if self.is_pairwise is True:
                _, q2, _, _, output_neg = self._create_inference(self.user_input_neg, self.item_input_neg, self.num_idx_neg)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function, self.result) + \
                            self.lambda_bilinear * l2_loss(p1) + \
                            self.gamma_bilinear * l2_loss(q2, q1, short) + \
                            self.reg_eta * l2_loss(eta_u, self.eta_bias)
            else:
                self.loss = learner.pointwise_loss(self.loss_function, self.labels, self.output) + \
                            self.lambda_bilinear * l2_loss(p1) + \
                            self.gamma_bilinear * l2_loss(q1, short) + \
                            self.reg_eta * l2_loss(eta_u, self.eta_bias)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
              
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        self.evaluate()
        for epoch in range(1, self.num_epochs+1):
            if self.is_pairwise is True:
                user_input_id, user_input, user_input_neg, num_idx_pos,\
                    num_idx_neg, item_input_pos, item_input_neg, item_input_recent = \
                    data_generator._get_pairwise_all_likefossil_data(self.dataset, self.high_order, self.train_dict)
                
                data_iter = DataIterator(user_input_id, user_input, user_input_neg, num_idx_pos, num_idx_neg,
                                         item_input_pos, item_input_neg, item_input_recent,
                                         batch_size=self.batch_size, shuffle=True)
            else:
                user_input_id, user_input, num_idx, item_input, item_input_recent, labels = \
                    data_generator._get_pointwise_all_likefossil_data(self.dataset, self.high_order,
                                                                      self.num_negatives, self.train_dict)

                data_iter = DataIterator(user_input_id, user_input, num_idx, item_input, item_input_recent, labels,
                                         batch_size=self.batch_size, shuffle=True)
           
            num_training_instances = len(user_input)
            total_loss = 0.0
            training_start_time = time()
            
            if self.is_pairwise is True:
                for bat_user_input_id, bat_users_pos, bat_users_neg, bat_idx_pos, bat_idx_neg, \
                        bat_items_pos, bat_items_neg, bat_item_input_recent in data_iter:
                    bat_users_pos = pad_sequences(bat_users_pos, value=self.num_items)
                    bat_users_neg = pad_sequences(bat_users_neg, value=self.num_items)
                    feed_dict = {self.user_input_id: bat_user_input_id,
                                 self.user_input: bat_users_pos,
                                 self.user_input_neg: bat_users_neg,
                                 self.num_idx: bat_idx_pos,
                                 self.num_idx_neg: bat_idx_neg,
                                 self.item_input: bat_items_pos,
                                 self.item_input_neg: bat_items_neg,
                                 self.item_input_recent: bat_item_input_recent}

                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
            else:
                for bat_user_input_id, bat_users, bat_idx, bat_items, bat_item_input_recent, bat_labels in data_iter:
                    bat_users = pad_sequences(bat_users, value=self.num_items)
                    feed_dict = {self.user_input_id: bat_user_input_id,
                                 self.user_input: bat_users,
                                 self.num_idx: bat_idx,
                                 self.item_input: bat_items,
                                 self.item_input_recent: bat_item_input_recent,
                                 self.labels: bat_labels}
                    
                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
                
            self.logger.info("[iter %d : loss : %f, time: %f]" %
                             (epoch, total_loss/num_training_instances, time()-training_start_time))
            
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)                

    def predict(self, user_ids, candidate_items_userids):
        ratings = []
        if candidate_items_userids is None:
            all_items = np.arange(self.num_items)
            for user_id in user_ids:
                items_by_user_id = self.train_dict[user_id]
                num_idx = len(items_by_user_id)
                # Get prediction scores
                item_idx = np.full(self.num_items, num_idx, dtype=np.int32)
                item_recent = []
                user_input = []
                user_input.extend([items_by_user_id]*self.num_items)
                for _ in range(self.num_items):
                    item_recent.append(items_by_user_id[len(items_by_user_id)-self.high_order:])
                users = np.full(self.num_items, user_id, dtype=np.int32)
                feed_dict = {self.user_input_id: users,
                             self.user_input: user_input,
                             self.num_idx: item_idx,
                             self.item_input: all_items,
                             self.item_input_recent: item_recent}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
                
        else:
            for user_id, eval_items_by_user_id in zip(user_ids, candidate_items_userids):
                items_by_user_id = self.train_dict[user_id]
                num_idx = len(items_by_user_id)
                # Get prediction scores
                item_idx = np.full(len(eval_items_by_user_id), num_idx, dtype=np.int32)
                user_input = []
                user_input.extend([items_by_user_id]*len(eval_items_by_user_id))
                item_recent = []
                for _ in range(len(eval_items_by_user_id)):
                    item_recent.append(items_by_user_id[len(items_by_user_id)-self.high_order:])
                users = np.full(len(eval_items_by_user_id), user_id, dtype=np.int32)
                feed_dict = {self.user_input_id: users,
                             self.user_input: user_input,
                             self.num_idx: item_idx,
                             self.item_input: eval_items_by_user_id,
                             self.item_input_recent: item_recent}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        return ratings
