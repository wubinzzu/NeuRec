from __future__ import absolute_import
from __future__ import division

import os
from model.AbstractRecommender import AbstractRecommender

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import logging
from time import time
from evaluation import Evaluate
import configparser
from util import learner,data_gen 
class FISM(AbstractRecommender):

    def __init__(self,sess,dataset):
        config = configparser.ConfigParser()
        config.read("conf/FISM.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("FISM arguments: %s " %(self.conf)) 
        self.verbose = int(self.conf["verbose"])
        self.batch_size = int(self.conf["batch_size"])
        self.num_epochs = int(self.conf["epochs"])
        self.embedding_size = int(self.conf["embedding_size"])
        self.regs = eval(self.conf["regs"])
        self.lambda_bilinear = self.regs[0]
        self.gamma_bilinear = self.regs[1]
        self.alpha = float(self.conf["alpha"]) 
        self.num_negatives= int(self.conf["num_neg"])
        self.learning_rate = float(self.conf["learning_rate"])
        self.learner = str(self.conf["learner"])
        self.topK = int(self.conf["topk"])
        self.loss_function = self.conf["loss_function"]
        self.ispairwise = self.conf["ispairwise"]
        self.num_negatives= int(self.conf["num_neg"])
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.dataset_name = dataset.dataset_name
        self.dataset = dataset
        self.sess=sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None], name = "user_input")    #the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None,],name = "num_idx")    #the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None,],name = "item_input_pos")      #the index of items
            if self.ispairwise.lower() =="true":
                self.user_input_neg = tf.placeholder(tf.int32, shape=[None, None], name = "user_input_neg")    
                self.item_input_neg = tf.placeholder(tf.int32, shape = [None,], name = "item_input_neg")
                self.num_idx_neg = tf.placeholder(tf.float32, shape=[None,],name = "num_idx_neg")
            else :
                self.lables = tf.placeholder(tf.float32, shape=[None,],name="labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            self.c1 = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                 name='c1', dtype=tf.float32)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2' )
            self.embedding_Q_ = tf.concat([self.c1,self.c2], 0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)
            self.bias = tf.Variable(tf.zeros(self.num_items),name='bias')

    def _create_inference(self,user_input,item_input,num_idx):
        with tf.name_scope("inference"):
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q_, user_input), 1)
            embedding_q = tf.nn.embedding_lookup(self.embedding_Q, item_input)
            bias_i = tf.nn.embedding_lookup(self.bias, item_input)
            coeff = tf.pow(num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            output = coeff * tf.reduce_sum(tf.multiply(embedding_p,embedding_q), 1) + bias_i
        return embedding_p, embedding_q,output 
    def _create_loss(self):
        with tf.name_scope("loss"):
            p1, q1, self.output = self._create_inference(self.user_input,self.item_input,self.num_idx)
            if self.ispairwise.lower() =="true":
                _, q2,output_neg = self._create_inference(self.user_input_neg,self.item_input_neg,self.num_idx_neg)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function,self.result) + self.lambda_bilinear * ( tf.reduce_sum(tf.square(p1))) \
                +self.gamma_bilinear*(tf.reduce_sum(tf.square(q2)) + tf.reduce_sum(tf.square(q1)))
            
            else:
                self.loss = learner.pointwise_loss(self.loss_function, \
                self.lables,tf.sigmoid(self.output)) + self.lambda_bilinear *\
                (tf.reduce_sum(tf.square(p1)))+self.gamma_bilinear *(tf.reduce_sum(tf.square(q1)))
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
              
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")

    def train_model(self):
        algo = "FISM"
        log_dir = "Log/%s/" % self.dataset_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        filename = log_dir+"log_{}_model_{}_lr_reg{}.txt".\
        format(algo,self.dataset_name,self.learning_rate,self.lambda_bilinear)
        
        logging.basicConfig(filename=filename, level=logging.INFO)
        logging.info("begin training %s model ......" % algo)
        logging.info(self.conf)
        for epoch in  range(self.num_epochs):
            if self.ispairwise.lower() =="true":
                user_input,user_input_neg, num_idx_pos, num_idx_neg, item_input_pos,item_input_neg = \
                data_gen._get_pairwise_all_likefism_data(self.dataset)
            else :
                user_input,num_idx,item_input,lables = data_gen._get_pointwise_all_likefism_data(self.dataset,self.num_negatives)
           
            num_training_instances = len(user_input)
            total_loss = 0.0
            training_start_time = time()
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                if self.ispairwise.lower() =="true":
                    bat_users_pos,bat_users_neg, bat_idx_pos, bat_idx_neg, \
                        bat_items_pos,bat_items_neg= \
                        data_gen._get_pairwise_batch_likefism_data(user_input,\
                        user_input_neg,self.dataset.num_items, num_idx_pos, num_idx_neg, item_input_pos,\
                        item_input_neg, num_batch, self.batch_size) 
                    feed_dict = {self.user_input:bat_users_pos,self.user_input_neg:bat_users_neg,\
                                self.num_idx:bat_idx_pos,self.num_idx_neg:bat_idx_neg,
                                self.item_input:bat_items_pos,self.item_input_neg:bat_items_neg}
                else :
                    bat_users,bat_idx,bat_items,bat_lables =\
                        data_gen._get_pointwise_batch_likefism_data(user_input,self.dataset.num_items,
                        num_idx,item_input,lables, num_batch, self.batch_size)
                    feed_dict = {self.user_input:bat_users,self.num_idx:bat_idx, self.item_input:bat_items,
                                self.lables:bat_lables}
    
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.valid_model(self,self.dataset,epoch)

    def predict(self, user_id,items,isvalid):
        if isvalid == True:
            cand_items = self.dataset.trainDict[user_id]
            num_idx = len(cand_items)
        else :
            cand_items = self.dataset.trainDict[user_id]
            if type(self.dataset.validDict[user_id]) == int:
                cand_items.append(self.dataset.validDict[user_id]) 
            else :
                cand_items.extend(self.dataset.validDict[user_id])
            num_idx = len(cand_items)
        # Get prediction scores
        item_idx = np.full(len(items), num_idx, dtype=np.int32)
        user_input = []
        for _ in range(len(items)):
            user_input.append(cand_items)
        feed_dict = {self.user_input: np.array(user_input), \
                     self.num_idx: item_idx, self.item_input:items}
        return self.sess.run((self.output), feed_dict=feed_dict)                