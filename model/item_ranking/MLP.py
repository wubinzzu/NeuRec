'''
Xiangnan He et al., "Neural Collaborative Filtering." in WWW 2017.
@author: WuBin
'''

from __future__ import absolute_import
from __future__ import division
import os
from model.AbstractRecommender import AbstractRecommender
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from time import time
from util import data_gen, learner
from evaluation import Evaluate
import configparser
class MLP(AbstractRecommender):
    def __init__(self,sess,dataset):  
        config = configparser.ConfigParser()
        config.read("conf/MLP.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("MLP arguments: %s " %(self.conf))
        self.layers = list(eval(self.conf["layers"]))
        self.learning_rate = float(self.conf["learning_rate"])
        self.ispairwise = self.conf["ispairwise"]
        self.learner = self.conf["learner"]
        self.topK = int(self.conf["topk"])
        self.num_epochs= int(self.conf["epochs"])
        self.num_negatives= int(self.conf["num_neg"])
        self.reg_mlp = float(self.conf["reg_mlp"])
        self.batch_size= int(self.conf["batch_size"])
        self.loss_function = self.conf["loss_function"]
        self.verbose= int(self.conf["verbose"])
        self.num_negatives= int(self.conf["num_neg"])
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items  
        self.dataset_name = dataset.dataset_name
        self.sess=sess                                       
        
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None,],name = 'user_input')
            self.item_input = tf.placeholder(tf.int32, shape=[None,],name = 'item_input')
            if self.ispairwise.lower() =="true":
                self.item_input_neg = tf.placeholder(tf.int32, shape = [None,], name = "item_input_neg")
            else :
                self.lables = tf.placeholder(tf.float32, shape=[None,],name="labels")
    
    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            self.mlp_embedding_user = tf.Variable(tf.random_normal(shape = [self.num_users,\
             int(self.layers[0]/2)],mean=0.0,stddev=0.01),name = "mlp_embedding_user",dtype=tf.float32)
            self.mlp_embedding_item = tf.Variable(tf.random_normal(shape = [self.num_items,\
             int(self.layers[0]/2)],mean=0.0,stddev=0.01),name = "mlp_embedding_item",dtype=tf.float32)
    def _create_inference(self,item_input):
        with tf.name_scope("inference"):
            # Crucial to flatten an embedding vector!
            mlp_user_latent = tf.nn.embedding_lookup(self.mlp_embedding_user,self.user_input)
            mlp_item_latent = tf.nn.embedding_lookup(self.mlp_embedding_item,item_input)   
            # The 0-th layer is the concatenation of embedding layers
            mlp_vector = tf.concat([mlp_user_latent, mlp_item_latent],axis=1)
            # MLP layers
            for idx in np.arange(len(self.layers)):
                mlp_vector = tf.layers.dense(mlp_vector,units=self.layers[idx], \
                activation=tf.nn.relu, name="layer%d" %idx)
                
            # Final prediction layer
        predict = tf.reduce_sum(mlp_vector,1)
        return  mlp_user_latent,mlp_item_latent,predict
             
    def _create_loss(self):
        with tf.name_scope("loss"):  
            p1, q1,self.output= self._create_inference(self.item_input)
            if self.ispairwise.lower() =="true":
                _, q2, self.output_neg = self._create_inference(self.item_input_neg)
                result = self.output - self.output_neg
                self.loss = learner.pairwise_loss(self.loss_function,result) + self.reg_mf * ( tf.reduce_sum(tf.square(p1)) \
                + tf.reduce_sum(tf.square(q2)) + tf.reduce_sum(tf.square(q1)))
                
            else :
                self.loss = learner.pointwise_loss(self.loss_function, self.lables,self.output)+self.reg_mlp * (tf.reduce_sum(tf.square(p1)) \
                   + tf.reduce_sum(tf.square(q1)))
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate) 
            
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
            
    def train_model(self):

        for epoch in  range(self.num_epochs):
            # Generate training instances
            if self.ispairwise.lower() =="true":
                user_input, item_input_pos, item_input_neg = data_gen._get_pairwise_all_data(self.dataset)
            else :
                user_input, item_input, lables = data_gen._get_pointwise_all_data(self.dataset, self.num_negatives)
            
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                if self.ispairwise.lower() =="true":
                    bat_users,bat_items_pos,bat_items_neg =\
                     data_gen._get_pairwise_batch_data(user_input,\
                     item_input_pos, item_input_neg, num_batch, self.batch_size)
                    feed_dict = {self.user_input:bat_users,self.item_input:bat_items_pos,\
                                self.item_input_neg:bat_items_neg}
                else:
                    bat_users, bat_items,bat_lables =\
                     data_gen._get_pointwise_batch_data(user_input, \
                     item_input, lables, num_batch, self.batch_size)
                    feed_dict = {self.user_input:bat_users, self.item_input:bat_items,
                                 self.lables:bat_lables}
                      
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)
                
    def predict(self, user_id, items):
        users = np.full(len(items), user_id, dtype=np.int32)
        return self.sess.run(self.output, feed_dict={self.user_input: users, self.item_input: items})  