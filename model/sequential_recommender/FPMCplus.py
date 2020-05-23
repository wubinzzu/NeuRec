"""
Reference: Steffen Rendle et al., "Factorizing Personalized Markov Chains
for Next-Basket Recommendation." in WWW 2010.
@author: wubin
"""
from model.AbstractRecommender import SeqAbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool
from util.tool import csr_to_user_dict_bytime
from util import timer
from util import l2_loss
from data import TimeOrderPairwiseSampler, TimeOrderPointwiseSampler


class FPMCplus(SeqAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(FPMCplus, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.embedding_size = conf["embedding_size"]
        self.weight_size = conf["weight_size"]
        self.learner = conf["learner"]
        self.loss_function = conf["loss_function"]
        self.is_pairwise = conf["is_pairwise"]
        self.num_epochs = conf["epochs"]
        self.reg_mf = conf["reg_mf"]
        self.reg_w = conf["reg_w"]
        self.batch_size = conf["batch_size"]
        self.high_order = conf["high_order"]
        self.verbose = conf["verbose"]
        self.embed_init_method = conf["embed_init_method"]
        self.weight_init_method = conf["weight_init_method"]
        self.stddev = float(conf["stddev"])
        self.num_negatives = conf["num_neg"]
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
            self.item_input_recent = tf.placeholder(tf.int32, shape=[None, None], name="item_input_recent")
            if self.is_pairwise is True:
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")
            else:
                self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            embed_initializer = tool.get_initializer(self.embed_init_method, self.stddev)
            weight_initializer = tool.get_initializer(self.weight_init_method, self.stddev)
            
            self.embeddings_UI = tf.Variable(embed_initializer([self.num_users, self.embedding_size]), 
                                             name='embeddings_UI', dtype=tf.float32)  # (users, embedding_size)
            self.embeddings_IU = tf.Variable(embed_initializer([self.num_items, self.embedding_size]),
                                             name='embeddings_IU', dtype=tf.float32)  # (items, embedding_size)
            self.embeddings_IL = tf.Variable(embed_initializer([self.num_items, self.embedding_size]),
                                             name='embeddings_IL', dtype=tf.float32)
            self.embeddings_LI = tf.Variable(embed_initializer([self.num_items, self.embedding_size]),
                                             name='embeddings_LI', dtype=tf.float32)  # (items, embedding_size)
            
            self.W = tf.Variable(weight_initializer([3*self.embedding_size, self.weight_size]),
                                 name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.b = tf.Variable(weight_initializer([1, self.weight_size]),
                                 name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)

    def _attention_mlp(self, embeddings_UI_u, embeddings_IL_i, item_embedding_recent):
        with tf.name_scope("attention_MLP"):
            UI_u = tf.tile(tf.expand_dims(embeddings_UI_u, 1), tf.stack([1, self.high_order, 1]))
            IL_i = tf.tile(tf.expand_dims(embeddings_IL_i, 1), tf.stack([1, self.high_order, 1]))
            embedding_short = tf.concat([UI_u,IL_i,item_embedding_recent], 2)
            batch_users = tf.shape(embedding_short)[0]

            # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            MLP_output = tf.matmul(tf.reshape(embedding_short, shape=[-1, 3*self.embedding_size]), self.W) + self.b
            MLP_output = tf.nn.tanh(MLP_output)

            # (b*n, w) * (w, 1) => (None, 1) => (b, n)
            A_ = tf.reshape(tf.matmul(MLP_output, self.h), shape=[batch_users, self.high_order])

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            exp_sum = tf.reduce_sum(exp_A_, 1, keepdims=True)  # (b, 1)

            A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)

            return tf.reduce_sum(A * item_embedding_recent, 1)    
            
    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            embeddings_UI_u = tf.nn.embedding_lookup(self.embeddings_UI, self.user_input)
            embeddings_IU_i = tf.nn.embedding_lookup(self.embeddings_IU, item_input)
            embeddings_IL_i = tf.nn.embedding_lookup(self.embeddings_IL, item_input)
            embeddings_LI_l = tf.nn.embedding_lookup(self.embeddings_LI, self.item_input_recent)
            item_embedding_short = self._attention_mlp(embeddings_UI_u, embeddings_IL_i, embeddings_LI_l)
            predict_vector = tf.multiply(embeddings_UI_u, embeddings_IU_i) + \
                             tf.multiply(embeddings_IL_i, item_embedding_short)
            predict = tf.reduce_sum(predict_vector, 1)
            return embeddings_UI_u, embeddings_IU_i, embeddings_IL_i, embeddings_LI_l, predict

    def _create_loss(self):
        with tf.name_scope("loss"):
            UI_u, IU_i, IL_i, LI_l, self.output = self._create_inference(self.item_input)
            if self.is_pairwise is True:
                _, IU_j, IL_j, _, output_neg = self._create_inference(self.item_input_neg)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function, self.result) + \
                            self.reg_mf * l2_loss(UI_u, IU_i, IL_i, LI_l, IU_j, IL_j) + \
                            self.reg_w * l2_loss(self.W, self.h)
            else:
                self.loss = learner.pointwise_loss(self.loss_function, self.labels, self.output) + \
                            self.reg_mf * l2_loss(UI_u, IU_i, IL_i, LI_l)

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
            data_iter = TimeOrderPairwiseSampler(self.dataset, high_order=self.high_order,
                                                 batch_size=self.batch_size, shuffle=True)
        else:
            data_iter = TimeOrderPointwiseSampler(self.dataset, high_order=self.high_order,
                                                  neg_num=self.num_negatives,
                                                  batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.num_epochs+1):
            num_training_instances = len(data_iter)
            total_loss = 0.0
            training_start_time = time()
      
            if self.is_pairwise is True:
                for bat_users, bat_items_recent, bat_items_pos, bat_items_neg in data_iter:
                    
                    feed_dict = {self.user_input: bat_users, 
                                 self.item_input: bat_items_pos,
                                 self.item_input_recent: bat_items_recent,
                                 self.item_input_neg: bat_items_neg}
                    
                    loss, _ = self.sess.run((self.loss,self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
            else:
                for bat_users, bat_items_recent, bat_items, bat_labels in data_iter:
                    
                    feed_dict = {self.user_input: bat_users,
                                 self.item_input: bat_items,
                                 self.item_input_recent: bat_items_recent,
                                 self.labels: bat_labels}
    
                    loss, _ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                    total_loss += loss
                
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/num_training_instances,
                                                             time()-training_start_time))
            
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)
                
    def predict(self, user_ids, candidate_items_user_ids):
        ratings = []
        if candidate_items_user_ids is None:
            all_items = np.arange(self.num_items)
            for user_id in user_ids:
                cand_items = self.train_dict[user_id]
                item_recent = []
    
                for _ in range(self.num_items):
                    item_recent.append(cand_items[len(cand_items)-self.high_order:])
                users = np.full(self.num_items, user_id, dtype=np.int32)
                feed_dict = {self.user_input: users,
                             self.item_input_recent: item_recent,
                             self.item_input: all_items}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))  
        
        else:
            for user_id, items_by_user_id in zip(user_ids, candidate_items_user_ids):
                cand_items = self.train_dict[user_id]
                item_recent = []
    
                for _ in range(len(items_by_user_id)):
                    item_recent.append(cand_items[len(cand_items)-self.high_order:])
                users = np.full(len(items_by_user_id), user_id, dtype=np.int32)
                feed_dict = {self.user_input: users,
                             self.item_input_recent: item_recent,
                             self.item_input: items_by_user_id}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        return ratings
