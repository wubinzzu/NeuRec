'''
Reference: Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." in WSDM2016
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
from util import learner,tool
from evaluation import Evaluate
import configparser
class CDAE(AbstractRecommender):
    def __init__(self,sess,dataset):  
        config = configparser.ConfigParser()
        config.read("conf/CDAE.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("CDAE arguments: %s " %(self.conf))
        self.hidden_neuron = int(self.conf["hidden_neuron"])
        self.learning_rate = float(self.conf["learning_rate"])
        self.learner = self.conf["learner"]
        self.loss_function = self.conf["loss_function"]
        self.topK = int(self.conf["topk"])
        self.reg = float(self.conf["reg"])
        self.num_epochs= int(self.conf["epochs"])
        self.batch_size= int(self.conf["batch_size"])
        self.verbose= int(self.conf["verbose"])
        self.h_act = self.conf["h_act"]
        self.g_act = self.conf["g_act"]
        self.corruption_level = float(self.conf["corruption_level"])
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items 
        self.dataset = dataset 
        self.sess=sess 
        
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None,],name = 'user_input')
            self.input_R = tf.placeholder(tf.float32, [None, self.num_items])
            self.mask_corruption = tf.placeholder(tf.float32, [None, self.num_items])
            
    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            self.V = tf.Variable(tf.random_normal([self.num_users,  self.hidden_neuron], stddev=0.01))
             
            self.weights = {
            'encoder': tf.Variable(tf.random_normal([self.num_items, self.hidden_neuron],stddev=0.01)),
            'decoder': tf.Variable(tf.random_normal([self.hidden_neuron, self.num_items],stddev=0.01)),
            }
            self.biases = {
            'encoder': tf.Variable(tf.random_normal([self.hidden_neuron],stddev=0.01)),
            'decoder': tf.Variable(tf.random_normal([self.num_items],stddev=0.01)),
            }
            
    def _create_inference(self):
        with tf.name_scope("inference"):
            
            self.user_latent =  tf.nn.embedding_lookup(self.V, self.user_input)
            
            corrupted_input = tf.multiply(self.input_R,self.mask_corruption)
            encoder_op = tool.activation_function(self.h_act,\
            tf.matmul(corrupted_input, self.weights['encoder'])+self.biases['encoder']+self.user_latent)
              
            self.decoder_op = tf.matmul(encoder_op, self.weights['decoder'])+self.biases['decoder']
            self.output = tool.activation_function(self.g_act,self.decoder_op)
            
    def _create_loss(self):
        with tf.name_scope("loss"):
            
            self.loss = learner.pointwise_loss(self.loss_function, self.input_R, self.output)
 
            self.reg_loss = self.reg*(tf.nn.l2_loss(self.weights['encoder'])+tf.nn.l2_loss(self.weights['decoder'])+
                tf.nn.l2_loss(self.biases['encoder'])+tf.nn.l2_loss(self.biases['decoder']))

            self.reg_loss = self.reg_loss + self.reg*tf.nn.l2_loss(self.user_latent)
            self.loss = self.loss + self.reg_loss
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
            mask_corruption_np = np.random.binomial(1, 1-self.corruption_level,
                                                (self.num_users, self.num_items))
            random_perm_doc_idx = np.random.permutation(self.num_users)
            self.total_batch = self.num_users
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = self.num_users
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                if num_batch == self.total_batch - 1:
                    batch_set_idx = random_perm_doc_idx[num_batch * self.batch_size:]
                elif num_batch < self.total_batch - 1:
                    batch_set_idx = random_perm_doc_idx[num_batch * self.batch_size: (num_batch + 1) * self.batch_size]
                
                batch_matrix = np.zeros((len(batch_set_idx),self.num_items)) 
                
                batch_uid = 0
                trainDict = self.dataset.trainDict
                for userid in batch_set_idx:
                    items_by_userid = trainDict[userid]
                    for itemid in items_by_userid:
                        batch_matrix[batch_uid,itemid] = 1
                        
                    batch_uid=batch_uid+1
                 
                feed_dict = feed_dict={self.mask_corruption: 
                    mask_corruption_np[batch_set_idx, :],\
                    self.input_R: batch_matrix, self.user_input: batch_set_idx}
                _, loss = self.sess.run([self.optimizer, self.loss],feed_dict=feed_dict)
                total_loss+=loss
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)
    
    def predict(self, user_id, items):
        mask = np.ones((1,self.num_items), dtype=np.int32)
        rating_matrix = np.zeros((1,self.num_items), dtype=np.int32)
        items_by_userid = self.dataset.trainDict[user_id]
        for itemid in items_by_userid:
            rating_matrix[0,itemid] = 1
        output = self.sess.run(self.output, feed_dict={self.mask_corruption:mask,self.input_R:rating_matrix,
            self.user_input: np.array([user_id])})
        return output[0,items]
    
    