'''
Reference: Steffen Rendle et al., "Factorizing Personalized Markov Chains
for Next-Basket Recommendation." in WWW 2010.
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
class FPMCplus(AbstractRecommender):
    properties = [
        "learning_rate",
        "embedding_size",
        "weight_size",
        "learner",
        "loss_function",
        "ispairwise",
        "topk",
        "epochs",
        "reg_mf",
        "reg_w",
        "batch_size",
        "high_order",
        "verbose",
        "num_neg"
    ]

    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.learning_rate = self.conf["learning_rate"]
        self.embedding_size = self.conf["embedding_size"]
        self.weight_size = self.conf["weight_size"]
        self.learner = self.conf["learner"]
        self.loss_function = self.conf["loss_function"]
        self.ispairwise = self.conf["ispairwise"]
        self.topK = self.conf["topk"]
        self.num_epochs= self.conf["epochs"]
        self.reg_mf = self.conf["reg_mf"]
        self.reg_w = self.conf["reg_w"]
        self.batch_size= self.conf["batch_size"]
        self.high_order = self.conf["high_order"]
        self.verbose= self.conf["verbose"]
        self.num_negatives= self.conf["num_neg"]
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None,], name = "user_input")
            self.item_input = tf.placeholder(tf.int32, shape = [None,], name = "item_input_pos")
            self.item_input_recents = tf.placeholder(tf.int32, shape = [None,None], name = "item_input_recent")
            if self.ispairwise == True:
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
            self.W = tf.Variable(tf.truncated_normal(shape=[3*self.embedding_size, self.weight_size],\
                mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.embedding_size))),\
                    name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, \
                stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.embedding_size))),name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)

    def _attention_MLP(self,embeddings_UI_u, embeddings_IL_i,item_embedding_recent):
        with tf.name_scope("attention_MLP"):
            UI_u = tf.tile(tf.expand_dims(embeddings_UI_u,1),tf.stack([1,self.high_order,1]))
            IL_i = tf.tile(tf.expand_dims(embeddings_IL_i,1),tf.stack([1,self.high_order,1]))
            embedding_short =tf.concat([UI_u,IL_i,item_embedding_recent],2)
            batch_users = tf.shape(embedding_short)[0]

            MLP_output = tf.matmul(tf.reshape(embedding_short,shape=[-1,3*self.embedding_size]), self.W) + self.b #(b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            MLP_output = tf.nn.tanh(MLP_output)

            A_ = tf.reshape(tf.matmul(MLP_output, self.h),shape=[batch_users,self.high_order]) #(b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            exp_sum = tf.reduce_sum(exp_A_, 1, keepdims=True)  # (b, 1)

            A = tf.expand_dims(tf.div(exp_A_, exp_sum),2) # (b, n, 1)

            return tf.reduce_sum(A * item_embedding_recent, 1)

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            embeddings_UI_u = tf.nn.embedding_lookup(self.embeddings_UI, self.user_input)
            embeddings_IU_i = tf.nn.embedding_lookup(self.embeddings_IU,item_input)
            embeddings_IL_i = tf.nn.embedding_lookup(self.embeddings_IL, item_input)
            embeddings_LI_l = tf.nn.embedding_lookup(self.embeddings_LI, self.item_input_recents)
            item_embedding_short = self._attention_MLP(embeddings_UI_u,embeddings_IL_i,embeddings_LI_l)
            predict_vector =  tf.multiply(embeddings_UI_u, embeddings_IU_i)+tf.multiply(embeddings_IL_i,item_embedding_short)
            predict = tf.reduce_sum(predict_vector, 1)
            return embeddings_UI_u,embeddings_IU_i, embeddings_IL_i,embeddings_LI_l,predict


    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            # loss for L(Theta)
            UI_u,IU_i,IL_i,LI_l,self.output = self._create_inference(self.item_input)
            if self.ispairwise == True:
                _, IU_j,IL_j,_,output_neg = self._create_inference(self.item_input_neg)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function,self.result) + self.reg_mf * ( tf.reduce_sum(tf.square(UI_u)) \
                + tf.reduce_sum(tf.square(IU_i)) + tf.reduce_sum(tf.square(IL_i)) + tf.reduce_sum(tf.square(LI_l))+ \
                 tf.reduce_sum(tf.square(LI_l))+tf.reduce_sum(tf.square(IU_j))+tf.reduce_sum(tf.square(IL_j)))+\
                 self.reg_w * (tf.reduce_sum(tf.square(self.W))+tf.reduce_sum(tf.square(self.h)))
            else :
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
        for epoch in  range(self.num_epochs):
            # Generate training instances
            if self.ispairwise == True:
                user_input, item_input_pos, item_input_recents, item_input_neg = \
                data_gen._get_pairwise_all_highorder_data(self.dataset,self.high_order)
            else :
                user_input, item_input,item_input_recents, lables = data_gen._get_pointwise_all_highorder_data(self.dataset,self.high_order, self.num_negatives)

            num_training_instances = len(user_input)
            total_loss = 0.0
            training_start_time = time()

            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                if self.ispairwise == True:
                    bat_users, bat_items_pos, bat_items_recents,bat_items_neg  = \
                    data_gen._get_pairwise_batch_seqdata(user_input, item_input_pos, \
                    item_input_recents, item_input_neg, num_batch, self.batch_size)
                    feed_dict = {self.user_input:bat_users,self.item_input:bat_items_pos,\
                                self.item_input_recents:bat_items_recents,self.item_input_neg:bat_items_neg}
                else :
                    bat_users, bat_items, bat_items_recents, bat_lables =\
                    data_gen._get_pointwise_batch_seqdata(user_input, \
                    item_input,item_input_recents, lables, num_batch, self.batch_size)
                    feed_dict = {self.user_input:bat_users, self.item_input:bat_items,
                                 self.item_input_recents:bat_items_recents,self.lables:bat_lables}

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
                                        self.item_input_recents:item_recents, self.item_input: items})
