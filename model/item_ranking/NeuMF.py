'''
Xiangnan He et al., "Neural Collaborative Filtering." in WWW 2017.
@author: WuBin
'''
from __future__ import absolute_import
from __future__ import division
import os
from model.AbstractRecommender import AbstractRecommender
from util.Logger import logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from time import time
from util import data_gen,learner
from evaluation import Evaluate
import configparser


class NeuMF(AbstractRecommender):
    def __init__(self,sess,dataset):  
        config = configparser.ConfigParser()
        config.read("conf/NeuMF.properties")
        self.conf=dict(config.items("hyperparameters"))
        # print("NeuMF arguments: %s " %(self.conf))
        self.embedding_size = int(self.conf["embedding_size"])
        self.layers = list(eval(self.conf["layers"]))
        self.reg_mf = float(self.conf["reg_mf"])
        self.reg_mlp = float(self.conf["reg_mlp"])
        self.learning_rate = float(self.conf["learning_rate"])
        self.learner = self.conf["learner"]
        self.loss_function = self.conf["loss_function"]
        self.topK = int(self.conf["topk"])
        self.num_epochs= int(self.conf["epochs"])
        self.num_negatives= int(self.conf["num_neg"])
        self.batch_size= int(self.conf["batch_size"])
        self.verbose= int(self.conf["verbose"])
        self.pretrain_epochs = int(self.conf["pretrain_epochs"])
        self.ispairwise = self.conf["ispairwise"]
        self.mf_pretrain = str(self.conf["mf_pretrain"])
        self.mlp_pretrain = str(self.conf["mlp_pretrain"])
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items 
        self.dataset = dataset 
        self.dataset_name = dataset.dataset_name
        self.sess=sess
        self.predict_output = None
        
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None,],name = 'user_input')
            self.item_input = tf.placeholder(tf.int32, shape=[None,],name = 'item_input')
            if self.ispairwise.lower() =="true":
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None,], name = "item_input_neg")
            else :
                self.lables = tf.placeholder(tf.float32, shape=[None,],name="labels")
            
    def _create_variables(self):
        with tf.name_scope("mf"):  # The embedding initialization is unknown now
            self.mf_embedding_user = tf.Variable(tf.random_normal(shape=[self.num_users,self.embedding_size],\
                mean=0.0,stddev=0.01),name = 'mf_embedding_user',dtype=tf.float32)
            self.mf_embedding_item = tf.Variable(tf.random_normal(shape=[self.num_items,self.embedding_size],\
                mean=0.0,stddev=0.01),name = 'mf_embedding_item',dtype=tf.float32)

            self.mf_output_layer = tf.layers.Dense(units=1, activation=tf.identity, name="mf_output_layer")

        with tf.name_scope("mlp"):
            self.mlp_embedding_user = tf.Variable(tf.random_normal(shape = [self.num_users,\
             int(self.layers[0]/2)],mean=0.0,stddev=0.01),name = "mlp_embedding_user",dtype=tf.float32)
            self.mlp_embedding_item = tf.Variable(tf.random_normal(shape = [self.num_items,\
             int(self.layers[0]/2)],mean=0.0,stddev=0.01),name = "mlp_embedding_item",dtype=tf.float32)

            self.mlp_hidden_layers = []
            for idx in np.arange(len(self.layers)):
                tmp_layer = tf.layers.Dense(units=self.layers[idx], activation=tf.nn.relu, name="mlp_layer_%d"%idx)
                self.mlp_hidden_layers.append(tmp_layer)
            self.mlp_output_layer = tf.layers.Dense(units=1, activation=tf.identity, name="mlp_output_layer")

        self.output_layer = tf.layers.Dense(units=1, activation=tf.identity, name="output_layer")

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            mf_user_latent = tf.nn.embedding_lookup(self.mf_embedding_user, self.user_input)
            mf_item_latent = tf.nn.embedding_lookup(self.mf_embedding_item, item_input)
            mlp_user_latent = tf.nn.embedding_lookup(self.mlp_embedding_user, self.user_input)
            mlp_item_latent = tf.nn.embedding_lookup(self.mlp_embedding_item, item_input)
            
            mf_vector = tf.multiply(mf_user_latent, mf_item_latent)  # element-wise multiply
            mf_output = self.mf_output_layer.apply(mf_vector)
            
            mlp_vector = tf.concat([mlp_user_latent, mlp_item_latent], axis=1)

            for layer in self.mlp_hidden_layers:
                mlp_vector = layer.apply(mlp_vector)

            mlp_output = self.mlp_output_layer.apply(mlp_vector)
    
            # Concatenate MF and MLP parts
            predict_vector = tf.concat([mf_vector, mlp_vector], axis=1)
            output = self.output_layer.apply(predict_vector)

            return mf_user_latent, mf_item_latent, mlp_user_latent, mlp_item_latent, tf.squeeze(output), tf.squeeze(mf_output), tf.squeeze(mlp_output)

    def _create_loss(self):
        with tf.name_scope("loss"):
            p1, q1, m1, n1, self.output, self.mf_output, self.mlp_output = self._create_inference(self.item_input)
            mf_reg_loss = tf.nn.l2_loss(p1)
            mf_reg_loss += tf.nn.l2_loss(q1)
            mlp_reg_loss = tf.nn.l2_loss(m1)
            mlp_reg_loss += tf.nn.l2_loss(n1)
            if self.ispairwise.lower() == "true":
                _, q2, _, n2, output_neg, mf_output_neg, mlp_output_neg = self._create_inference(self.item_input_neg)
                result = self.output - output_neg
                # Regularization loss
                mf_reg_loss += tf.nn.l2_loss(q2)
                mlp_reg_loss += tf.nn.l2_loss(n2)
                # NeuMF loss
                loss = learner.pairwise_loss(self.loss_function, result)
                # GMF loss
                mf_loss = learner.pairwise_loss(self.loss_function, self.mf_output-mf_output_neg)
                # MLP loss
                mlp_loss = learner.pairwise_loss(self.loss_function, self.mlp_output-mlp_output_neg)

            else:
                # NeuMF loss
                loss = learner.pointwise_loss(self.loss_function, self.lables, self.output)
                # GMF loss
                mf_loss = learner.pointwise_loss(self.loss_function, self.lables, self.mf_output)
                # MLP loss
                mlp_loss = learner.pointwise_loss(self.loss_function, self.lables, self.mlp_output)

            # Total loss
            self.loss = loss + self.reg_mf*mf_reg_loss + self.reg_mlp*mlp_reg_loss
            self.mf_loss = mf_loss + self.reg_mf*mf_reg_loss
            self.mlp_loss = mlp_loss + self.reg_mlp*mlp_reg_loss

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)

            mf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mf')
            self.mf_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mf_loss, var_list=mf_vars)
            mlp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mlp')
            self.mlp_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mlp_loss, var_list=mlp_vars)
            
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()

    def _pre_training(self):
        logger.info("pre-training...")
        for epoch in range(self.pretrain_epochs):
            # Generate training instances
            if self.ispairwise.lower() == "true":
                user_input, item_input_pos, item_input_neg = data_gen._get_pairwise_all_data(self.dataset)
            else:
                user_input, item_input, lables = data_gen._get_pointwise_all_data(self.dataset, self.num_negatives)

            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for num_batch in np.arange(int(num_training_instances / self.batch_size)):
                if self.ispairwise.lower() == "true":
                    bat_users, bat_items_pos, bat_items_neg = \
                        data_gen._get_pairwise_batch_data(user_input, item_input_pos, item_input_neg, num_batch, self.batch_size)
                    feed_dict = {self.user_input: bat_users,
                                 self.item_input: bat_items_pos,
                                 self.item_input_neg: bat_items_neg}
                else:
                    bat_users, bat_items, bat_lables = \
                        data_gen._get_pointwise_batch_data(user_input, item_input, lables, num_batch, self.batch_size)
                    feed_dict = {self.user_input: bat_users,
                                 self.item_input: bat_items,
                                 self.lables: bat_lables}

                self.sess.run((self.mf_optimizer, self.mlp_optimizer), feed_dict=feed_dict)
            if epoch % self.verbose == 0:
                logger.info("GMF test")
                self.predict_output = self.mf_output
                Evaluate.test_model(self, self.dataset)

                logger.info("MLP test")
                self.predict_output = self.mlp_output
                Evaluate.test_model(self, self.dataset)

    def train_model(self):
        self._pre_training()
        logger.info("NeuMF training...")
        self.predict_output = self.output
        for epoch in range(self.num_epochs):
            # Generate training instances
            if self.ispairwise.lower() == "true":
                user_input, item_input_pos, item_input_neg = data_gen._get_pairwise_all_data(self.dataset)
            else :
                user_input, item_input, lables = data_gen._get_pointwise_all_data(self.dataset, self.num_negatives)
            
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                if self.ispairwise.lower() == "true":
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
        return self.sess.run(self.predict_output, feed_dict={self.user_input: users, self.item_input: items})
