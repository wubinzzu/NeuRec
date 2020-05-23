"""
Xiangnan He et al., "Neural Collaborative Filtering." in WWW 2017.
@author: WuBin
"""
from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool
from util import timer
import pickle
from util import l2_loss
from data import PairwiseSampler, PointwiseSampler


class NeuMF(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(NeuMF, self).__init__(dataset, conf)
        self.embedding_size = conf["embedding_size"]
        self.layers = conf["layers"]
        self.reg_mf = conf["reg_mf"]
        self.reg_mlp = conf["reg_mlp"]
        self.learning_rate = conf["learning_rate"]
        self.learner = conf["learner"]
        self.loss_function = conf["loss_function"]
        self.num_epochs = conf["epochs"]
        self.num_negatives = conf["num_neg"]
        self.batch_size = conf["batch_size"]
        self.verbose = conf["verbose"]
        self.is_pairwise = conf["is_pairwise"]
        self.mf_pretrain = conf["mf_pretrain"]
        self.mlp_pretrain = conf["mlp_pretrain"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items 
        self.dataset = dataset 
        self.sess = sess
        
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None], name='user_input')
            self.item_input = tf.placeholder(tf.int32, shape=[None], name='item_input')
            if self.is_pairwise is True:
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")
            else:
                self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")
            
    def _create_variables(self, params=None):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            if params is None:
                initializer = tool.get_initializer(self.init_method, self.stddev)
                
                self.mf_embedding_user = tf.Variable(initializer([self.num_users, self.embedding_size]),
                                                     name='mf_embedding_user', dtype=tf.float32)
                self.mf_embedding_item = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                                     name='mf_embedding_item', dtype=tf.float32)
                self.mlp_embedding_user = tf.Variable(initializer([self.num_users, int(self.layers[0]/2)]),
                                                      name="mlp_embedding_user", dtype=tf.float32)
                self.mlp_embedding_item = tf.Variable(initializer([self.num_items, int(self.layers[0]/2)]),
                                                      name="mlp_embedding_item", dtype=tf.float32)
            else:
                self.mf_embedding_user = tf.Variable(params[0][0], name='mf_embedding_user', dtype=tf.float32)
                self.mf_embedding_item = tf.Variable(params[0][1], name='mf_embedding_item', dtype=tf.float32)
                
                self.mlp_embedding_user = tf.Variable(params[1][0], name="mlp_embedding_user", dtype=tf.float32)
                self.mlp_embedding_item = tf.Variable(params[1][1], name="mlp_embedding_item", dtype=tf.float32)

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            
            mf_user_latent = tf.nn.embedding_lookup(self.mf_embedding_user, self.user_input)
            mf_item_latent = tf.nn.embedding_lookup(self.mf_embedding_item, item_input)
            mlp_user_latent = tf.nn.embedding_lookup(self.mlp_embedding_user, self.user_input)
            mlp_item_latent = tf.nn.embedding_lookup(self.mlp_embedding_item, item_input)
            
            mf_vector = tf.multiply(mf_user_latent, mf_item_latent)  # element-wise multiply
            
            mlp_vector = tf.concat([mlp_user_latent, mlp_item_latent], axis=1)

            for idx in np.arange(len(self.layers)):
                mlp_vector = tf.layers.dense(mlp_vector, units=self.layers[idx], activation=tf.nn.relu)
    
            # Concatenate MF and MLP parts
            predict = tf.reduce_sum(tf.concat([mf_vector, mlp_vector], axis=1), 1)
            return mf_user_latent, mf_item_latent, mlp_user_latent, mlp_item_latent, predict

    def _create_loss(self):
        with tf.name_scope("loss"):
            p1, q1, m1, n1, self.output = self._create_inference(self.item_input)
            if self.is_pairwise is True:
                _, q2, _, n2, output_neg = self._create_inference(self.item_input_neg)
                result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function, result) + \
                            self.reg_mf * l2_loss(p1, q2, q1) + \
                            self.reg_mlp * l2_loss(m1, n2, n1)
            else:
                self.loss = learner.pointwise_loss(self.loss_function, self.labels, self.output) + \
                            self.reg_mf * l2_loss(p1, q1) + \
                            self.reg_mlp * l2_loss(m1, n1)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
            
    def build_graph(self):
        self._create_placeholders()
        try:
            pre_trained_params = []
            with open(self.mf_pretrain, "rb") as fin:
                pre_trained_params.append(pickle.load(fin, encoding="utf-8"))
            with open(self.mlp_pretrain, "rb") as fin:
                pre_trained_params.append(pickle.load(fin, encoding="utf-8"))
            self.logger.info("load pretrained params successful!")
        except:
            pre_trained_params = None
            self.logger.info("load pretrained params unsuccessful!")
            
        self._create_variables(pre_trained_params)
        self._create_loss()
        self._create_optimizer()
                                               
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        if self.is_pairwise is True:
            data_iter = PairwiseSampler(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True)
        else:
            data_iter = PointwiseSampler(self.dataset, neg_num=self.num_negatives, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.num_epochs+1):
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(data_iter)
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
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss / num_training_instances,
                                                             time() - training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_user_ids):
        ratings = []
        if candidate_items_user_ids is not None:
            for u, i in zip(user_ids, candidate_items_user_ids):
                users = np.full(len(i), u, dtype=np.int32)
                feed_dict = {self.user_input: users, self.item_input: i}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        else:
            for u in user_ids:
                users = np.full(self.num_items, u, dtype=np.int32)
                feed_dict = {self.user_input: users, self.item_input: np.arange(self.num_items)}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        return ratings
