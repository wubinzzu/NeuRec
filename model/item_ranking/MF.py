'''
Reference: Steffen Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
    GMF: Xiangnan He et al., "Neural Collaborative Filtering." in WWW 2017.
@author: wubin
'''
from __future__ import absolute_import
from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from time import time
import configparser
from util import learner,data_gen
from evaluation import Evaluate
from model.AbstractRecommender import AbstractRecommender
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class MF(AbstractRecommender):
    def __init__(self,sess,dataset):  
        config = configparser.ConfigParser()
        config.read("conf/MF.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("MF arguments: %s " %(self.conf))
        self.learning_rate = float(self.conf["learning_rate"])
        self.embedding_size = int(self.conf["embedding_size"])
        self.learner = self.conf["learner"]
        self.loss_function = self.conf["loss_function"]
        self.ispairwise = self.conf["ispairwise"]
        self.topK = int(self.conf["topk"])
        self.num_epochs= int(self.conf["epochs"])
        self.reg_mf = float(self.conf["reg_mf"])
        self.batch_size= int(self.conf["batch_size"])
        self.verbose= int(self.conf["verbose"])
        self.num_negatives= int(self.conf["num_neg"])
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.sess=sess  
    
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None,], name = "user_input")
            self.item_input = tf.placeholder(tf.int32, shape = [None,], name = "item_input")
            if self.ispairwise.lower() =="true":
                self.item_input_neg = tf.placeholder(tf.int32, shape = [None,], name = "item_input_neg")
            else :
                self.lables = tf.placeholder(tf.float32, shape=[None,],name="labels")
    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.user_embeddings = tf.Variable(tf.random_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='user_embeddings', dtype=tf.float32)  #(users, embedding_size)
            self.item_embeddings = tf.Variable(tf.random_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='item_embeddings', dtype=tf.float32)  #(items, embedding_size)
    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, item_input)
            predict_vector = tf.multiply(user_embedding, item_embedding)
            return user_embedding, item_embedding, predict_vector

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            p1, q1,predict_vector= self._create_inference(self.item_input)
            if self.ispairwise.lower() =="true":
                self.output = tf.reduce_sum(predict_vector,1)
                _, q2, predict_vector_neg = self._create_inference(self.item_input_neg)
                self.output_neg = tf.reduce_sum(predict_vector_neg,1)
                result = self.output - self.output_neg
                self.loss = learner.pairwise_loss(self.loss_function,result) + self.reg_mf * ( tf.reduce_sum(tf.square(p1)) \
                + tf.reduce_sum(tf.square(q2)) + tf.reduce_sum(tf.square(q1)))
                
            else :
                prediction = tf.layers.dense(inputs=predict_vector,units=1, activation=tf.nn.sigmoid)
                self.output = tf.squeeze(prediction)
                self.loss = learner.pointwise_loss(self.loss_function, self.lables,self.output)+self.reg_mf * (tf.reduce_sum(tf.square(p1)) \
                   + tf.reduce_sum(tf.square(q1)))

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
                #Evaluate.test_model(self,self.dataset)
                
    def predict(self, user_id, items):
        users = np.full(len(items), user_id, dtype=np.int32)
        return self.sess.run(self.output, feed_dict={self.user_input: users, self.item_input: items})  