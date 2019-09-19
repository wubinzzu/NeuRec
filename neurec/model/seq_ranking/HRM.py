'''
Reference: Pengfei Wang et al., "Learning Hierarchical Representation Model for NextBasket Recommendation." in SIGIR 2015.
@author: wubin
'''
from __future__ import absolute_import
from __future__ import division
import os
from neurec.model.AbstractRecommender import AbstractRecommender
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from time import time
from neurec.util import learner, data_gen, reader
from neurec.evaluation import Evaluate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class HRM(AbstractRecommender):
    def __init__(self,sess,dataset):  
        self.conf = reader.config("HRM.properties", "hyperparameters")

        print("HRM arguments: %s " %(self.conf))
        self.learning_rate = float(self.conf["learning_rate"])
        self.embedding_size = int(self.conf["embedding_size"])
        self.learner = self.conf["learner"]
        self.topK = int(self.conf["topk"])
        self.num_epochs= int(self.conf["epochs"])
        self.reg_mf = float(self.conf["reg_mf"])
        self.pre_agg= self.conf["pre_agg"]
        self.loss_function = self.conf["loss_function"]
        self.session_agg= self.conf["session_agg"]
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
            self.user_input = tf.compat.v1.placeholder(tf.int32, shape = [None,], name = "user_input")
            self.item_input = tf.compat.v1.placeholder(tf.int32, shape = [None,], name = "item_input_pos")
            self.item_input_recent = tf.compat.v1.placeholder(tf.int32, shape = [None,None], name = "item_input_recent")
            self.lables = tf.compat.v1.placeholder(tf.float32, shape=[None,],name="labels")
    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='user_embeddings', dtype=tf.float32)  #(users, embedding_size)
            self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='item_embeddings', dtype=tf.float32)  #(items, embedding_size)
    
    def avg_pooling(self, imput_embedding):
        output = tf.reduce_mean(imput_embedding, axis=1)
        return output

    def max_pooling(self, imput_embedding):
        output = tf.reduce_max(imput_embedding, axis=1)
        return output
    
    def _create_inference(self):
        with tf.name_scope("inference"):
            # embedding look up
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            item_embedding_recent = tf.nn.embedding_lookup(self.item_embeddings, self.item_input_recent)
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)
            if self.high_order>1:
                if self.session_agg == "max":
                    item_embedding_short = self.max_pooling(item_embedding_recent)  
                elif self.session_agg == "avg":
                    item_embedding_short = self.avg_pooling(item_embedding_recent) 
                concat_user_item = tf.concat([tf.expand_dims(user_embedding,1),tf.expand_dims(item_embedding_short,1)],axis=1)    
            else:
                concat_user_item = tf.concat([tf.expand_dims(user_embedding,1),tf.expand_dims(item_embedding_recent,1)],axis=1) 
            if self.pre_agg == "max":
                hybrid_user_embedding = self.max_pooling(concat_user_item)
            elif self.pre_agg == "avg":
                hybrid_user_embedding = self.avg_pooling(concat_user_item)
            predict_vector = tf.multiply(hybrid_user_embedding, item_embedding)
            predict = tf.reduce_sum(predict_vector,1)
            return user_embedding, item_embedding,item_embedding_recent,predict
            

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            p1,q1,r1,self.output = self._create_inference()
            self.loss = learner.pointwise_loss(self.loss_function,self.lables,self.output) + self.reg_mf * (tf.reduce_sum(tf.square(p1)) \
            +tf.reduce_sum(tf.square(r1))+ tf.reduce_sum(tf.square(q1)))

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
                             self.item_input_recent:bat_items_recents,self.lables:bat_lables}
    
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
                                        self.item_input_recent:item_recents, self.item_input: items})  
