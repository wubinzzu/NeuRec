"""
Reference: Dawen, Liang, et al. "Variational autoencoders for collaborative filtering." in WWW2018
@author: wubin
"""
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
from model.AbstractRecommender import AbstractRecommender
from util import timer
from util.tool import csr_to_user_dict


class MultiVAE(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(MultiVAE, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.learner = conf["learner"]
        self.batch_size = conf["batch_size"]
        self.act = conf["activation"]
        self.reg = conf["reg"]
        self.num_epochs = conf["epochs"]
        self.anneal_cap = conf["anneal_cap"]
        self.total_anneal_steps = conf["total_anneal_steps"]
        self.weight_init_method = conf["weight_init_method"]
        self.bias_init_method = conf["bias_init_method"]
        self.stddev = conf["stddev"]
        self.verbose = conf["verbose"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items  
        self.p_dims = conf["p_dim"] + [self.num_items]
        self.q_dims = self.p_dims[::-1]
        self.dims = self.q_dims + self.p_dims[1:]
        self.train_dict = csr_to_user_dict(dataset.train_matrix)
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
            self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)
            self.is_training_ph = tf.placeholder_with_default(0., shape=None)
            self.anneal_ph = tf.placeholder_with_default(1., shape=None)

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now   
            self.weights_q, self.biases_q = [], []
            weight_initializer = tool.get_initializer(self.weight_init_method, self.stddev)
            bias_initializer = tool.get_initializer(self.bias_init_method, self.stddev)
            for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
                if i == len(self.q_dims[:-1]) - 1:
                    # we need two sets of parameters for mean and variance,
                    # respectively
                    d_out *= 2
                weight_key = "weight_q_{}to{}".format(i, i+1)
                bias_key = "bias_q_{}".format(i+1)
                
                self.weights_q.append(tf.Variable(weight_initializer([d_in, d_out]), name=weight_key, dtype=tf.float32))
                
                self.biases_q.append(tf.Variable(bias_initializer([d_out]), name=bias_key, dtype=tf.float32))

            self.weights_p, self.biases_p = [], []
    
            for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
                weight_key = "weight_p_{}to{}".format(i, i+1)
                bias_key = "bias_p_{}".format(i+1)
                
                self.weights_p.append(tf.Variable(weight_initializer([d_in, d_out]), name=weight_key, dtype=tf.float32))
                
                self.biases_p.append(tf.Variable(bias_initializer([d_out]), name=bias_key, dtype=tf.float32))

    def q_graph(self):
        mu_q, std_q, KL = None, None, None
        
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights_q) - 1:
                h = tool.activation_function(self.act, h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]  # log sigmod^2  batch x 200

                std_q = tf.exp(0.5 * logvar_q)  # sigmod batch x 200
                KL = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar_q + tf.exp(logvar_q) + tf.pow(mu_q,2) - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        self.h = z
        
        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            self.h = tf.matmul(self.h, w) + b
            
            if i != len(self.weights_p) - 1:
                self.h = tool.activation_function(self.act, self.h)
        return self.h

    def _create_inference(self):
        with tf.name_scope("inference"):
            # q-network
            mu_q, std_q, self.KL = self.q_graph()
            epsilon = tf.random_normal(tf.shape(std_q), mean=0, stddev=0.01)
    
            sampled_z = mu_q + self.is_training_ph * epsilon * std_q  # batch x 200
    
            # p-network
            logits = self.p_graph(sampled_z)
            
            self.log_softmax_var = tf.nn.log_softmax(logits)
               
    def _create_loss(self):
        with tf.name_scope("loss"):  
            neg_ll = -tf.reduce_mean(tf.reduce_sum(self.log_softmax_var * self.input_ph, axis=1))
            # apply regularization to weights
            reg = l2_regularizer(self.reg)
            
            reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
            # tensorflow l2 regularization multiply 0.5 to the l2 norm
            # multiply 2 so that it is back in the same scale
            self.loss = neg_ll + self.anneal_ph * self.KL + 2 * reg_var  # neg_ELBO
                
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
        update_count = 0.0
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(1,self.num_epochs+1):
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
                
                batch_matrix = np.zeros((len(batch_set_idx), self.num_items))
                
                if self.total_anneal_steps > 0:
                    anneal = min(self.anneal_cap, 1. * update_count / self.total_anneal_steps)
                else:
                    anneal = self.anneal_cap
                
                batch_uid = 0
                for user_id in batch_set_idx:
                    items_by_user_id = self.train_dict[user_id]
                    for item_id in items_by_user_id:
                        batch_matrix[batch_uid, item_id] = 1
                        
                    batch_uid = batch_uid+1
                 
                feed_dict = {self.input_ph: batch_matrix,
                             self.keep_prob_ph: 0.8,
                             self.anneal_ph: anneal,
                             self.is_training_ph: 1}
                _, loss = self.sess.run([self.optimizer, self.loss],feed_dict=feed_dict)
                total_loss += loss
                
                update_count += 1
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/num_training_instances,
                                                             time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_user_ids):
        ratings = []
        if candidate_items_user_ids is not None:
            rating_matrix = np.zeros((1, self.num_items), dtype=np.int32)
            for user_id, candidate_items_user_id in zip(user_ids, candidate_items_user_ids):
                items_by_user_id = self.dataset.train_matrix[user_id].indices
                for item_id in items_by_user_id:
                    rating_matrix[0, item_id] = 1
                output = self.sess.run(self.h, feed_dict={self.input_ph: rating_matrix})
                ratings.append(output[0, candidate_items_user_id])
                
        else:
            rating_matrix = np.zeros((1, self.num_items), dtype=np.int32)
            all_items = np.arange(self.num_items)
            for user_id in user_ids:
                items_by_user_id = self.dataset.train_matrix[user_id].indices
                for item_id in items_by_user_id:
                    rating_matrix[0, item_id] = 1
                output = self.sess.run(self.h, feed_dict={self.input_ph: rating_matrix})
                ratings.append(output[0, all_items])
        return ratings
