'''
Reference: Xiangnan He et al., "NAIS: Neural Attentive Item Similarity Model for Recommendation." in TKDE2018
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
from evaluation import Evaluate
import configparser
from util import data_gen,learner
class NAIS(AbstractRecommender):
    def __init__(self,sess,dataset):
        config = configparser.ConfigParser()
        config.read("conf/NAIS.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("NAIS arguments: %s " %(self.conf)) 
        self.pretrain = int(self.conf["pretrain"])
        self.verbose = int(self.conf["verbose"])
        self.batch_size = int(self.conf["batch_size"])
        self.num_epochs = int(self.conf["epochs"])
        self.weight_size = int(self.conf["weight_size"])
        self.embedding_size = int(self.conf["embedding_size"])
        self.data_alpha = float(self.conf["data_alpha"])
        self.regs = eval(self.conf["regs"])
        self.ispairwise = self.conf["ispairwise"]
        self.topK = int(self.conf["topk"])
        self.lambda_bilinear = self.regs[0]
        self.gamma_bilinear = self.regs[1]
        self.eta_bilinear = self.regs[2] 
        self.alpha = float(self.conf["alpha"]) 
        self.beta = float(self.conf["beta"])
        self.num_negatives= int(self.conf["num_neg"])
        self.learning_rate = float(self.conf["learning_rate"])
        self.activation = self.conf["activation"]
        self.loss_function = self.conf["loss_function"]
        self.algorithm = int(self.conf["algorithm"])
        self.learner = str(self.conf["learner"])
        self.dataset = dataset
        self.dataset_name= dataset.dataset_name 
        self.num_items = dataset.num_items
        self.num_users = dataset.num_users
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
            trainable_flag = (self.pretrain!=2)
            self.c1 = tf.Variable(tf.truncated_normal(shape=[self.num_items,\
                self.embedding_size], mean=0.0, stddev=0.01), name='c1', \
                    dtype=tf.float32, trainable=trainable_flag)
            
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], axis=0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items,\
                self.embedding_size], mean=0.0, stddev=0.01),name='embedding_Q',\
                dtype=tf.float32,trainable=trainable_flag)
            
            self.bias = tf.Variable(tf.zeros(self.num_items),name='bias',trainable=trainable_flag)

            # Variables for attention
            if self.algorithm == 0:
                self.W = tf.Variable(tf.truncated_normal(shape=[self.embedding_size,\
                 self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size\
                 + self.embedding_size))),name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            else:    
                self.W = tf.Variable(tf.truncated_normal(shape=[2*self.embedding_size,\
                self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size\
                + 2*self.embedding_size))),name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size],\
                mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.embedding_size))),\
                name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)
            
    def _create_inference(self,user_input,item_input,num_idx):
        with tf.name_scope("inference"):
            embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, user_input) # (b, n, e)
            embedding_q = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_Q,item_input),1) # (b, 1, e)
            
            if self.algorithm == 0:
                embedding_p = self._attention_MLP(embedding_q_ * embedding_q,embedding_q_,num_idx)
            else:
                n = tf.shape(user_input)[1]
                embedding_p = self._attention_MLP(tf.concat([embedding_q_,\
                     tf.tile(embedding_q, tf.stack([1,n,1]))],2),embedding_q_,num_idx)

            embedding_q = tf.reduce_sum(embedding_q, 1)
            bias_i = tf.nn.embedding_lookup(self.bias, item_input)
            coeff = tf.pow(num_idx, tf.constant(self.alpha, tf.float32, [1]))
            output = coeff *tf.reduce_sum(embedding_p*embedding_q, 1)+ bias_i
            
            return embedding_q_,embedding_q,output
    
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

    def _attention_MLP(self, q_,embedding_q_,num_idx):
            with tf.name_scope("attention_MLP"):
                b = tf.shape(q_)[0]
                n = tf.shape(q_)[1]
                r = (self.algorithm + 1)*self.embedding_size
    
                MLP_output = tf.matmul(tf.reshape(q_,[-1,r]), self.W) + self.b #(b*n, e or 2*e) * (e or 2*e, w) + (1, w)
                if self.activation == 0:
                    MLP_output = tf.nn.relu( MLP_output )
                elif self.activation == 1:
                    MLP_output = tf.nn.sigmoid( MLP_output )
                elif self.activation == 2:
                    MLP_output = tf.nn.tanh( MLP_output )
    
                A_ = tf.reshape(tf.matmul(MLP_output, self.h),[b,n]) #(b*n, w) * (w, 1) => (None, 1) => (b, n)
    
                # softmax for not mask features
                exp_A_ = tf.exp(A_)
                mask_mat = tf.sequence_mask(num_idx, maxlen = n, dtype = tf.float32) # (b, n)
                exp_A_ = mask_mat * exp_A_
                exp_sum = tf.reduce_sum(exp_A_, 1, keepdims=True)  # (b, 1)
                exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))
    
                A = tf.expand_dims(tf.div(exp_A_, exp_sum),2) # (b, n, 1)
    
                return tf.reduce_sum(A * embedding_q_, 1)      

    def train_model(self):

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
                Evaluate.test_model(self,self.dataset)

    def predict(self, user_id,items):
        cand_items = self.dataset.trainDict[user_id]
        num_idx = len(cand_items)
        # Get prediction scores
        item_idx = np.full(len(items), num_idx, dtype=np.int32)
        user_input = []
        for _ in range(len(items)):
            user_input.append(cand_items)
        feed_dict = {self.user_input: np.array(user_input), \
                     self.num_idx: item_idx, self.item_input:items}
        return self.sess.run((self.output), feed_dict=feed_dict)     