from __future__ import absolute_import
from __future__ import division
import os
from model.AbstractRecommender import AbstractRecommender
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import logging
from time import time
from util import learner, data_gen
from evaluation import Evaluate
import configparser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class FPMC(AbstractRecommender):
    def __init__(self,sess,dataset):  
        config = configparser.ConfigParser()
        config.read("conf/FPMC.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("FPMC arguments: %s " %(self.conf))
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
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.dataset = dataset
        self.sess=sess  
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None,], name = "user_input")
            self.item_input = tf.placeholder(tf.int32, shape = [None,], name = "item_input_pos")
            self.item_input_recent = tf.placeholder(tf.int32, shape = [None,], name = "item_input_recent")
            if self.ispairwise.lower() =="true":
                self.item_input_neg = tf.placeholder(tf.int32, shape = [None,], name = "item_input_neg")
            else :
                self.lables = tf.placeholder(tf.float32, shape=[None,],name="labels")
    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embeddings_UI = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embeddings_UI', dtype=tf.float32)  #(users, embedding_size)
            self.embeddings_IU = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embeddings_IU', dtype=tf.float32)  #(items, embedding_size)
            self.embeddings_IL = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embeddings_IL', dtype=tf.float32) 
            self.embeddings_LI = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embeddings_LI', dtype=tf.float32)  #(items, embedding_size) 
            
    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            embeddings_UI_u = tf.nn.embedding_lookup(self.embeddings_UI, self.user_input)
            embeddings_IU_i = tf.nn.embedding_lookup(self.embeddings_IU,item_input)
            embeddings_IL_i = tf.nn.embedding_lookup(self.embeddings_IL, item_input)
            embeddings_LI_l = tf.nn.embedding_lookup(self.embeddings_LI, self.item_input_recent)
            return embeddings_UI_u, embeddings_IU_i,embeddings_IL_i,embeddings_LI_l,\
                tf.multiply(embeddings_UI_u, embeddings_IU_i) + tf.multiply(embeddings_IL_i, embeddings_LI_l)
                #tf.multiply(user_embedding, item_embedding)+item_embedding_short

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            UI_u,IU_i,IL_i,LI_l,predict_vector = self._create_inference(self.item_input)
            if self.ispairwise.lower() =="true":
                self.output = tf.reduce_sum(predict_vector,1)
                _, IU_j,IL_j,_,predict_vector_neg = self._create_inference(self.item_input_neg)
                output_neg = tf.reduce_sum(predict_vector_neg,1)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function,self.result) + self.reg_mf * ( tf.reduce_sum(tf.square(UI_u)) \
                + tf.reduce_sum(tf.square(IU_i)) + tf.reduce_sum(tf.square(IL_i)) + tf.reduce_sum(tf.square(LI_l))+ \
                 tf.reduce_sum(tf.square(LI_l))+tf.reduce_sum(tf.square(IU_j))+tf.reduce_sum(tf.square(IL_j)))
            else :
                prediction = tf.layers.dense(inputs=predict_vector,units=1, activation=tf.nn.sigmoid)
                self.output = tf.squeeze(prediction)
                self.loss = learner.pointwise_loss(self.loss_function,self.lables,self.output) + self.reg_mf * (tf.reduce_sum(tf.square(UI_u)) \
                +tf.reduce_sum(tf.square(IU_i))+ tf.reduce_sum(tf.square(IL_i))+tf.reduce_sum(tf.square(LI_l)))

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
        algo = "FPMC"
    
        log_dir = "Log/%s/" % self.dataset.dataset_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        filename = log_dir+"log_{}_model_{}_lr_reg{}.txt".\
        format(algo,self.dataset.dataset_name,self.learning_rate,self.reg_mf)
        
        logging.basicConfig(filename=filename, level=logging.INFO)
        logging.info("begin training %s model ......" % algo)
        logging.info(self.conf)
        for epoch in  range(self.num_epochs):
            # Generate training instances
            if self.ispairwise.lower() =="true":
                user_input, item_input_pos, item_input_recent, item_input_neg = \
                data_gen._get_pairwise_all_firstorder_data(self.dataset)
            else :
                user_input, item_input,item_input_recent, lables = data_gen._get_pointwise_all_firstorder_data(self.dataset,self.num_negatives)
           
            num_training_instances = len(user_input)
            total_loss = 0.0
            training_start_time = time()
      
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                if self.ispairwise.lower() =="true":
                    bat_users, bat_items_pos, bat_items_recents,bat_items_neg  = \
                    data_gen._get_pairwise_batch_seqdata(user_input, item_input_pos, \
                    item_input_recent, item_input_neg, num_batch, self.batch_size) 
                    feed_dict = {self.user_input:bat_users,self.item_input:bat_items_pos,\
                                self.item_input_recent:bat_items_recents,self.item_input_neg:bat_items_neg}
                else :
                    bat_users, bat_items, bat_items_recents, bat_lables =\
                    data_gen._get_pointwise_batch_seqdata(user_input, \
                    item_input,item_input_recent, lables, num_batch, self.batch_size)
                    feed_dict = {self.user_input:bat_users, self.item_input:bat_items,
                                 self.item_input_recent:bat_items_recents,self.lables:bat_lables}
    
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
                
            logging.info("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)
                #Evaluate.test_model(self,self.dataset)
    def predict(self, user_id, items):
        cand_items = self.dataset.trainDict[user_id]
        item_recent = np.full(len(items), cand_items[-1], dtype='int32')

        users = np.full(len(items), user_id, dtype='int32')
        return self.sess.run((self.output), feed_dict={self.user_input: users,\
                self.item_input_recent:item_recent, self.item_input: items})  