"""
Reference: Ruining He et al., "Translation-based Recommendation." in RecSys 2017
@author: wubin
"""

from model.AbstractRecommender import SeqAbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool
from util.data_iterator import DataIterator
from util.tool import csr_to_user_dict_bytime
from util import timer
from util import l2_loss
from data import TimeOrderPointwiseSampler, TimeOrderPairwiseSampler


def l2_distance(a, b, name="euclidean_distance"):
    return tf.norm(a - b, ord='euclidean', axis=-1, name=name)


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
            if self.is_pairwise is True:
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")
            else:
                self.labels = tf.placeholder(tf.float32, shape=[None, ], name="labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            initializer = tool.get_initializer(self.init_method, self.stddev)
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
            predict_vector = user_embedding + tf.tile(self.global_embedding, tf.stack([batch_size, 1])) + \
                             item_embedding_recent - item_embedding
            predict = item_bias-tf.reduce_sum(tf.square(predict_vector), 1)
            return user_embedding, item_embedding_recent, item_embedding, item_bias, predict

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            p1, r1, q1, b1, self.output = self._create_inference(self.item_input)
            if self.is_pairwise is True:
                _, _, q2, b2, output_neg = self._create_inference(self.item_input_neg)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function, self.result) + \
                            self.reg_mf * l2_loss(p1, r1, q2, q1, b1, b2, self.global_embedding)
            else:
                self.loss = learner.pointwise_loss(self.loss_function, self.labels, self.output) + \
                            self.reg_mf * l2_loss(p1, r1, q1, b1, self.global_embedding)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
                
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        u_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
        last_i_emb = tf.nn.embedding_lookup(self.item_embeddings, self.item_input_recent)
        pre_emb = u_emb + self.global_embedding + last_i_emb
        pre_emb = tf.expand_dims(pre_emb, axis=1)  # b*1*d
        j_emb = tf.expand_dims(self.item_embeddings, axis=0)  # 1*n*d
        self.prediction = -l2_distance(pre_emb, j_emb) + self.item_biases  # b*n

    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        if self.is_pairwise is True:
            data_iter = TimeOrderPairwiseSampler(self.dataset, high_order=1, neg_num=1,
                                                 batch_size=self.batch_size, shuffle=True)
        else:
            data_iter = TimeOrderPointwiseSampler(self.dataset, high_order=1,
                                                  neg_num=self.num_negatives,
                                                  batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            num_training_instances = len(data_iter)
            total_loss = 0.0
            training_start_time = time()

            if self.is_pairwise is True:
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

            # logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss / num_training_instances,
            #                                                  time() - training_start_time))
            
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, items=None):
        users = DataIterator(user_ids, batch_size=64, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            last_items = [self.train_dict[u][-1] for u in bat_user]
            feed = {self.user_input: bat_user, self.item_input_recent: last_items}
            bat_ratings = self.sess.run(self.prediction, feed_dict=feed)
            all_ratings.append(bat_ratings)
        all_ratings = np.vstack(all_ratings)

        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]

        return all_ratings
