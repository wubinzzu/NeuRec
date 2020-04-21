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


class MultiDAE(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(MultiDAE, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.learner = conf["learner"]
        self.batch_size = conf["batch_size"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items  
        self.p_dims = conf["p_dim"] + [self.num_items]
        self.q_dims = self.p_dims[::-1]
        self.dims = self.q_dims + self.p_dims[1:]
        self.act = conf["activation"]
        self.reg = conf["reg"]
        self.num_epochs = conf["epochs"]
        self.weight_init_method = conf["weight_init_method"]
        self.bias_init_method = conf["bias_init_method"]
        self.stddev = conf["stddev"]
        self.verbose = conf["verbose"]
        self.train_dict = csr_to_user_dict(dataset.train_matrix)
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
            self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now   
            self.weights = []
            self.biases = []
            weight_initializer = tool.get_initializer(self.weight_init_method, self.stddev)
            bias_initializer = tool.get_initializer(self.bias_init_method, self.stddev)
            # define weights
            for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
                weight_key = "weight_{}to{}".format(i, i+1)
                bias_key = "bias_{}".format(i+1)
                
                self.weights.append(tf.Variable(weight_initializer([d_in, d_out]), name=weight_key, dtype=tf.float32))
                
                self.biases.append(tf.Variable(bias_initializer([d_out]), name=bias_key, dtype=tf.float32))
    
    def _create_inference(self):
        with tf.name_scope("inference"):
            # construct forward graph        
            self.h = tf.nn.l2_normalize(self.input_ph, 1)
            self.h = tf.nn.dropout(self.h, self.keep_prob_ph)
            
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                self.h = tf.matmul(self.h, w) + b
                
                if i != len(self.weights) - 1:
                    self.h = tool.activation_function(self.act, self.h)
                    
            self.log_softmax_var = tf.nn.log_softmax(self.h)
        
    def _create_loss(self):
        with tf.name_scope("loss"):  
            # per-user average negative log-likelihood 
            neg_ll = -tf.reduce_mean(tf.reduce_sum(self.log_softmax_var * self.input_ph, axis=1))
            # apply regularization to weights
            regularization = l2_regularizer(self.reg)
            reg_var = apply_regularization(regularization, self.weights)
            # tensorflow l2 regularization multiply 0.5 to the l2 norm
            # multiply 2 so that it is back in the same scale
            self.loss = neg_ll + 2 * reg_var   
                
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

        for epoch in range(1, self.num_epochs+1):
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
                
                batch_uid = 0
                for user_id in batch_set_idx:
                    items_by_user_id = self.train_dict[user_id]
                    for item_id in items_by_user_id:
                        batch_matrix[batch_uid, item_id] = 1
                        
                    batch_uid = batch_uid+1
                 
                feed_dict = {self.input_ph: batch_matrix, self.keep_prob_ph: 0.5}
                _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                total_loss += loss
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
                output = self.sess.run(self.h, feed_dict={self.input_ph:rating_matrix})
                ratings.append(output[0, candidate_items_user_id])
                
        else:
            rating_matrix = np.zeros((1, self.num_items), dtype=np.int32)
            all_items = np.arange(self.num_items)
            for user_id in user_ids:
                items_by_user_id = self.dataset.train_matrix[user_id].indices
                for item_id in items_by_user_id:
                    rating_matrix[0, item_id] = 1
                output = self.sess.run(self.h, feed_dict={self.input_ph:rating_matrix})
                ratings.append(output[0, all_items])
        return ratings
