'''
Reference: Ruining He et al., "Translation-based Recommendation." in RecSys 2017
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
from neurec.util import learner, data_gen
from neurec.evaluation import Evaluate
from neurec.util.properties import Properties
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TransRec(AbstractRecommender):
    properties = [
        "learning_rate",
        "embedding_size",
        "learner",
        "loss_function",
        "ispairwise",
        "topk",
        "epochs",
        "reg_mf",
        "batch_size",
        "verbose",
        "num_neg"
    ]

    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.learning_rate = self.conf["learning_rate"]
        self.embedding_size = self.conf["embedding_size"]
        self.learner = self.conf["learner"]
        self.loss_function = self.conf["loss_function"]
        self.ispairwise = self.conf["ispairwise"]
        self.topK = self.conf["topk"]
        self.num_epochs= self.conf["epochs"]
        self.reg_mf = self.conf["reg_mf"]
        self.batch_size= self.conf["batch_size"]
        self.verbose= self.conf["verbose"]
        self.num_negatives= self.conf["num_neg"]
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None,], name = "user_input")
            self.item_input = tf.placeholder(tf.int32, shape = [None,], name = "item_input_pos")
            self.item_input_recent = tf.placeholder(tf.int32, shape = [None,], name = "item_input_recent")
            if self.ispairwise == True:
                self.item_input_neg = tf.placeholder(tf.int32, shape = [None,], name = "item_input_neg")
            else :
                self.lables = tf.placeholder(tf.float32, shape=[None,],name="labels")
    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='user_embeddings', dtype=tf.float32)  #(users, embedding_size)
            self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='item_embeddings', dtype=tf.float32)  #(items, embedding_size)
            self.item_biases = tf.Variable(tf.truncated_normal(shape=[self.num_items], mean=0.0, stddev=0.01),
                name='item_biases', dtype=tf.float32)  #(items)
            self.global_embedding = tf.Variable(tf.truncated_normal(shape=[1,self.embedding_size], mean=0.0, stddev=0.01),
                name='global_embedding', dtype=tf.float32)

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            batch_size =  tf.shape(user_embedding)[0]
            item_embedding_recent = tf.nn.embedding_lookup(self.item_embeddings, self.item_input_recent)
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, item_input)
            item_bias = tf.nn.embedding_lookup(self.item_biases, item_input)
            predict_vector = user_embedding + tf.tile(self.global_embedding, tf.stack([batch_size,1]))+item_embedding_recent -item_embedding
            predict = item_bias-tf.reduce_sum(tf.square(predict_vector),1)
            return user_embedding, item_embedding_recent,item_embedding,item_bias,predict

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            p1,r1,q1,b1,self.output = self._create_inference(self.item_input)
            if self.ispairwise == True:
                _,_,q2,b2,output_neg = self._create_inference(self.item_input_neg)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function,self.result) + self.reg_mf * ( tf.reduce_sum(tf.square(p1)) \
                +tf.reduce_sum(tf.square(r1)) + tf.reduce_sum(tf.square(q2)) + tf.reduce_sum(tf.square(q1))+tf.reduce_sum(tf.square(b1)) + tf.reduce_sum(tf.square(b2)) + tf.reduce_sum(tf.square(self.global_embedding)))
            else :
                self.loss = learner.pointwise_loss(self.loss_function,self.lables,self.output) + self.reg_mf * (tf.reduce_sum(tf.square(p1)) \
                +tf.reduce_sum(tf.square(r1))+ tf.reduce_sum(tf.square(q1))+ tf.reduce_sum(tf.square(b1))+ tf.reduce_sum(tf.square(self.global_embedding)))

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
            if self.ispairwise == True:
                user_input, item_input_pos, item_input_recents, item_input_neg = \
                data_gen._get_pairwise_all_firstorder_data(self.dataset)
            else :
                user_input, item_input,item_input_recents, lables = data_gen._get_pointwise_all_firstorder_data(self.dataset,self.num_negatives)


            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                if self.ispairwise == True:
                    bat_users, bat_items_pos, bat_items_recents,bat_items_neg  = \
                    data_gen._get_pairwise_batch_seqdata(user_input, item_input_pos, \
                    item_input_recents, item_input_neg, num_batch, self.batch_size)
                    feed_dict = {self.user_input:bat_users,self.item_input:bat_items_pos,\
                                self.item_input_recent:bat_items_recents,self.item_input_neg:bat_items_neg}
                else :
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
        item_recent = np.full(len(items), cand_items[-1], dtype='int32')

        users = np.full(len(items), user_id, dtype='int32')
        return self.sess.run((self.output), feed_dict={self.user_input: users,\
                                        self.item_input_recent:item_recent, self.item_input: items})
