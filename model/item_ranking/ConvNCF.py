'''
Xiangnan He et al., "Outer Product-based Neural Collaborative Filtering", In IJCAI 2018.  
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
from util import data_gen
class ConvNCF(AbstractRecommender):
    def __init__(self,sess,dataset):  
        config = configparser.ConfigParser()
        config.read("conf/ConvNCF.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("ConvNCF arguments: %s " %(self.conf))
        self.embedding_size = int(self.conf["embedding_size"])
        self.topK = int(self.conf["topk"])
        self.regs = eval(self.conf["regs"])
        self.lambda_bilinear = float( self.regs[0])
        self.gamma_bilinear = float( self.regs[1])
        self.lambda_weight = float( self.regs[2])
        self.keep = float(self.conf["keep"])
        self.num_epochs= int(self.conf["epochs"])
        self.batch_size= int(self.conf["batch_size"])
        self.nc=eval(self.conf["net_channel"])
        self.lr_embed=float(self.conf["lr_embed"])
        self.lr_net=float(self.conf["lr_net"])
        self.verbose= int(self.conf["verbose"])
        self.loss_function = self.conf["loss_function"]
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items 
        self.dataset = dataset
        self.sess=sess 
        
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None,], name = "user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape = [None,], name = "item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape = [None,], name = "item_input_neg")
            self.keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
    #---------- model definition -------
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv_weight(self, isz, osz):
        return (self.weight_variable([2,2,isz,osz]), self.bias_variable([osz]))

    def _conv_layer(self, input, P):
        conv = tf.nn.conv2d(input, P[0], strides=[1, 2, 2, 1], padding='SAME')
        return tf.nn.tanh(conv + P[1])

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.1),
                                                                name='embedding_P', dtype=tf.float32)  #(users, embedding_size)
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.1),
                                                                name='embedding_Q', dtype=tf.float32)  #(items, embedding_size)

            # here should have 6 iszs due to the size of outer products is 64x64
            iszs = [1] + self.nc[:-1]
            oszs = self.nc
            self.P = []
            for isz, osz in zip(iszs, oszs):
                self.P.append(self._conv_weight(isz, osz))

            self.W = self.weight_variable([self.nc[-1], 1])#32x1
            self.b = self.weight_variable([1])#1

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            self.embedding_p = tf.nn.embedding_lookup(self.embedding_P, self.user_input)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, item_input)

            # outer product of P_u and Q_i
            self.relation = tf.matmul(tf.expand_dims(self.embedding_p,2), tf.expand_dims(self.embedding_q,1))
            self.net_input = tf.expand_dims(self.relation, -1)

            # CNN
            self.layer = []
            input = self.net_input
            for p in self.P:
                self.layer.append(self._conv_layer(input, p))
                input = self.layer[-1]
            # prediction
            self.dropout = tf.nn.dropout(self.layer[-1], self.keep_prob)
            self.output_layer = tf.matmul(tf.reshape(self.dropout,[-1,self.nc[-1]]), self.W) + self.b

            return self.embedding_p, self.embedding_q, self.output_layer

    def _regular(self, params):
        res = 0
        for param in params:
            res += tf.reduce_sum(tf.square(param[0])) + tf.reduce_sum(tf.square(param[1]))
        return res

    def _create_loss(self):
        with tf.name_scope("loss"):
            # BPR loss for L(Theta)
            self.p1, self.q1, self.output = self._create_inference(self.item_input_pos)
            self.p2, self.q2, self.output_neg = self._create_inference(self.item_input_neg)
            self.result = self.output - self.output_neg
            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result)))

            self.opt_loss = self.loss + self.lambda_bilinear * ( tf.reduce_sum(tf.square(self.p1)) \
                                    + tf.reduce_sum(tf.square(self.q2)) + tf.reduce_sum(tf.square(self.q1)))\
                                    + self.gamma_bilinear * self._regular([(self.W, self.b)]) \
                                    + self.lambda_weight * (self._regular(self.P) + self._regular([(self.W, self.b)]))

    # used at the first time when emgeddings are pretrained yet network are randomly initialized
    # if not, the parameters may be NaN.
    def _create_pre_optimizer(self):
        self.pre_opt = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.loss)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
        # seperated optimizer
            var_list1 = [self.embedding_P, self.embedding_Q]
            #[self.W1,self.W2,self.W3,self.W4,self.b1,self.b2,self.b3,self.b4,self.P1,self.P2,self.P3]
            var_list2 = list(set(tf.trainable_variables()) - set(var_list1))
            opt1 = tf.train.AdagradOptimizer(self.lr_embed)
            opt2 = tf.train.AdagradOptimizer(self.lr_net)
            grads = tf.gradients(self.opt_loss, var_list1 + var_list2)
            grads1 = grads[:len(var_list1)]
            grads2 = grads[len(var_list1):]
            train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
            train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
            self.optimizer = tf.group(train_op1, train_op2)   
            
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_pre_optimizer()
        self._create_optimizer()                             
    #---------- training process -------
    def train_model(self):
        for epoch in  range(self.num_epochs):
            # Generate training instances
            user_input, item_input_pos, item_input_neg = data_gen._get_pairwise_all_data(self.dataset)
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                bat_users,bat_items_pos,bat_items_neg =\
                 data_gen._get_pairwise_batch_data(user_input,\
                 item_input_pos, item_input_neg, num_batch, self.batch_size)
                feed_dict = {self.user_input:bat_users,self.item_input_pos:bat_items_pos,\
                            self.item_input_neg:bat_items_neg,self.keep_prob:0.8}
                #out_put, out_put_neg = self.sess.run((self.output, self.output_neg), feed_dict=feed_dict)
                #print("out_put:\t", out_put)
                #print("out_put_neg:\t", out_put_neg)
                      
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset,epoch)
                
    def predict(self, user_id, items):
        users = np.full(len(items), user_id, dtype=np.int32)
        return self.sess.run(self.output, feed_dict={self.user_input: users, self.item_input_pos: items,self.keep_prob:1.0})