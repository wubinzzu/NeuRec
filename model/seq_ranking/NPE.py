'''
Reference: ThaiBinh Nguyen, et al. "NPE: Neural Personalized Embedding for Collaborative Filtering" in ijcai2018
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
from util import learner, data_gen
from evaluation import Evaluate
import configparser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class NPE(AbstractRecommender):
    def __init__(self,sess,dataset):  
        config = configparser.ConfigParser()
        config.read("conf/NPE.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("NPE arguments: %s " %(self.conf))
        self.learning_rate = float(self.conf["learning_rate"])
        self.embedding_size = int(self.conf["embedding_size"])
        self.learner = self.conf["learner"]
        self.loss_function = self.conf["loss_function"]
        self.topK = int(self.conf["topk"])
        self.num_epochs= int(self.conf["epochs"])
        self.reg = float(self.conf["reg"])
        self.batch_size= int(self.conf["batch_size"])
        self.high_order = int(self.conf["high_order"])
        self.verbose= int(self.conf["verbose"])
        self.num_negatives= int(self.conf["num_neg"])
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.dataset = dataset
        self.sess=sess  
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None,], name = "user_input")
            self.item_input = tf.placeholder(tf.int32, shape = [None,], name = "item_input")
            self.item_input_recents = tf.placeholder(tf.int32, shape = [None,None], name = "item_input_recents")
            self.lables = tf.placeholder(tf.float32, shape=[None,],name="labels")
    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embeddings_UI = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embeddings_UI', dtype=tf.float32)  #(users, embedding_size)
            self.embeddings_IU = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embeddings_IU', dtype=tf.float32)  #(items, embedding_size)
            self.embeddings_IL = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embeddings_IL', dtype=tf.float32) 
    def _create_inference(self):
        with tf.name_scope("inference"):
            # embedding look up
            embeddings_UI_u = tf.nn.embedding_lookup(self.embeddings_UI, self.user_input)
            embeddings_IU_i = tf.nn.embedding_lookup(self.embeddings_IU,self.item_input)
            embeddings_LI_l = tf.nn.embedding_lookup(self.embeddings_IL, self.item_input_recents)
            
            context_embedding = tf.reduce_sum(embeddings_LI_l,1)
            return embeddings_UI_u,embeddings_IU_i,embeddings_LI_l,\
               tf.multiply(tf.nn.relu(embeddings_UI_u), tf.nn.relu(embeddings_IU_i))\
               +tf.multiply(tf.nn.relu(embeddings_IU_i),tf.nn.relu(context_embedding))

    def _create_loss(self):
        with tf.name_scope("loss"):
            UI_u,IU_i,LI_l,predict_vector = self._create_inference()
            prediction = tf.layers.dense(inputs=predict_vector,units=1, activation=tf.nn.sigmoid)
            self.output = tf.squeeze(prediction)
            self.loss = learner.pointwise_loss(self.loss_function,self.lables,self.output) + self.reg* (tf.reduce_sum(tf.square(UI_u)) \
            +tf.reduce_sum(tf.square(IU_i))+tf.reduce_sum(tf.square(LI_l)))

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
                
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
#---------- training process -------
    def train_model(self):
        for epoch in  range(self.num_epochs):
            # Generate training instances
            user_input, item_input,item_input_recents, lables = data_gen._get_pointwise_all_highorder_data(self.dataset,self.high_order, self.num_negatives)
           
            num_training_instances = len(user_input)
            total_loss = 0.0
            training_start_time = time()
      
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
               
                bat_users, bat_items, bat_items_recents, bat_lables =\
                data_gen._get_pointwise_batch_seqdata(user_input, \
                item_input,item_input_recents, lables, num_batch, self.batch_size)
                feed_dict = {self.user_input:bat_users, self.item_input:bat_items,
                             self.item_input_recents:bat_items_recents,self.lables:bat_lables}
    
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
                
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)
    def predict(self, user_id, items):
        cand_items = self.dataset.trainDict[user_id]
        item_recents = []
        for _ in range(len(items)):
            item_recents.append( cand_items[len(cand_items)-self.high_order:])
        users = np.full(len(items), user_id, dtype='int32')
        return self.sess.run((self.output), feed_dict={self.user_input: users,\
                                        self.item_input_recents:item_recents, self.item_input: items})  