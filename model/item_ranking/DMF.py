'''
Reference: Hong-Jian Xue et al., "Deep Matrix Factorization Models for Recommender Systems." In IJCAI2017.  
@author: wubin
'''
from __future__ import absolute_import
from __future__ import division
import os
from model.AbstractRecommender import AbstractRecommender
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from time import time
from util import learner
from evaluation import Evaluate
import configparser
class DMF(AbstractRecommender):
    def __init__(self,sess,dataset):  
        config = configparser.ConfigParser()
        config.read("conf/DMF.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("DMF arguments: %s " %(self.conf))
        self.embedding_size = int(self.conf["embedding_size"])
        self.learning_rate = float(self.conf["learning_rate"])
        self.learner = self.conf["learner"]
        self.topK = int(self.conf["topk"])
        self.num_epochs= int(self.conf["epochs"])
        self.num_negatives= int(self.conf["num_neg"])
        self.batch_size= int(self.conf["batch_size"])
        self.verbose= int(self.conf["verbose"])
        self.layers = list(eval(self.conf["layers"]))
        self.loss_function = self.conf["loss_function"]
        self.fist_layer_size = self.layers[0]
        self.last_layer_size = self.layers[1]
        self.neg_sample_size = self.num_negatives
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items 
        self.dataset = dataset 
        self.user_matrix = self.dataset.trainMatrix.tocsr()
        self.item_matrix = self.dataset.trainMatrix.tocsc()
        self.dataset_name = dataset.dataset_name
        self.sess=sess 
        
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.one_hot_u = tf.placeholder(tf.float32, shape=[None,None],name = 'user_input')
            self.one_hot_v = tf.placeholder(tf.float32, shape=[None,None],name = 'item_input')
            self.lables = tf.placeholder(tf.float32, shape=[None,],name="labels")
            
    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            self.u_w1 = tf.get_variable("u_w1", shape=(self.num_items, self.fist_layer_size), initializer=tf.contrib.layers.xavier_initializer())
            self.u_b1 = tf.get_variable("u_b1", shape=[self.fist_layer_size], initializer=tf.contrib.layers.xavier_initializer())
            self.u_w2 = tf.get_variable("u_w2", shape=(self.fist_layer_size, self.last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
            self.u_b2 = tf.get_variable("u_b2", shape=[self.last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
        
            self.v_w1 = tf.get_variable("v_w1", shape=(self.num_users, self.fist_layer_size), initializer=tf.contrib.layers.xavier_initializer())
            self.v_b1 = tf.get_variable("v_b1", shape=[self.fist_layer_size], initializer=tf.contrib.layers.xavier_initializer())
            self.v_w2 = tf.get_variable("v_w2", shape=(self.fist_layer_size, self.last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
            self.v_b2 = tf.get_variable("v_b2", shape=[self.last_layer_size], initializer=tf.contrib.layers.xavier_initializer())

    def _create_inference(self):
        with tf.name_scope("inference"):
            net_u_1 = tf.nn.relu(tf.matmul(self.one_hot_u, self.u_w1) + self.u_b1)
            net_u_2 = tf.matmul(net_u_1, self.u_w2) + self.u_b2
        
            net_v_1 = tf.nn.relu(tf.matmul(self.one_hot_v, self.v_w1) + self.v_b1)
            net_v_2 = tf.matmul(net_v_1, self.v_w2) + self.v_b2
        
            fen_zhi = tf.reduce_sum(net_u_2 * net_v_2, 1)
        
            norm_u = tf.reduce_sum(tf.square(net_u_2), 1)
            norm_v = tf.reduce_sum(tf.square(net_v_2), 1)
            fen_mu = norm_u * norm_v
            self.output =  tf.nn.relu(fen_zhi / fen_mu)
            
    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = - tf.reduce_sum(
            self.lables * tf.log(self.output+ 1e-10) + (1 - self.lables) * tf.log(1 - self.output+ 1e-10)) 
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate) 
            
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
                                               
    def train_model(self):

        for epoch in  range(self.num_epochs):
            # Generate training instances
            user_input, item_input, lables = self._get_input_all_data()
            
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                num_training_instances =len(user_input) 
                id_start = num_batch * self.batch_size
                id_end = (num_batch + 1) *  self.batch_size
                if id_end>num_training_instances:
                    id_end=num_training_instances
                bat_users = user_input[id_start:id_end].tolist()
                bat_items = item_input[id_start:id_end].tolist()
                bat_lables = np.array(lables[id_start:id_end])
                feed_dict = {self.one_hot_u:bat_users, self.one_hot_v:bat_items,
                             self.lables:bat_lables}
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)
    def _get_input_all_data(self):
        user_input,item_input,lables = [],[],[]
        for u in range(self.num_users):
            # positive instance
            items_by_user = self.user_matrix[u].indices
            u_vector = np.reshape(self.user_matrix.getrow(u).toarray(),[self.num_items])
            for i in items_by_user:
                i_vector = np.reshape(self.item_matrix.getcol(i).toarray(),[self.num_users])
                user_input.append(u_vector)
                item_input.append(i_vector)
                lables.append(1)
            # negative instance
                for _ in range(self.num_negatives):
                    j = np.random.randint(self.num_items)
                    while (u, j) in self.dataset.trainMatrix.keys():
                        j = np.random.randint(self.num_items)
                    j_vector = np.reshape(self.item_matrix.getcol(i).toarray(),[self.num_users])
                    user_input.append(u_vector)
                    item_input.append(j_vector)
                    lables.append(0)
        user_input = np.array(user_input, dtype=np.int32)
        item_input = np.array(item_input, dtype=np.int32)
        lables = np.array(lables, dtype=np.float32)
        num_training_instances = len(user_input)
        shuffle_index = np.arange(num_training_instances,dtype=np.int32)
        np.random.shuffle(shuffle_index)
        user_input=user_input[shuffle_index]
        item_input=item_input[shuffle_index]
        lables = lables[shuffle_index]
        return user_input,item_input,lables            
    def predict(self, user_id, items):
        user_input,item_input = [],[]
        u_vector = np.reshape(self.user_matrix.getrow(user_id).toarray(),[self.num_items])
        for i in items:
            user_input.append(u_vector)
            i_vector = np.reshape(self.item_matrix.getcol(i).toarray(),[self.num_users])
            item_input.append(i_vector)
        return self.sess.run(self.output, feed_dict={self.one_hot_u: user_input, self.one_hot_v: item_input})