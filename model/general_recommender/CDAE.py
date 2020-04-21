"""
Reference: Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." in WSDM2016
@author: wubin
"""
from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool
from util import timer
from util import l2_loss
from util.data_iterator import DataIterator


class CDAE(AbstractRecommender):
    def __init__(self, sess, dataset, conf):  
        super(CDAE, self).__init__(dataset, conf)
        self.hidden_neuron = conf["hidden_neuron"]
        self.learning_rate = conf["learning_rate"]
        self.learner = conf["learner"]
        self.reg = conf["reg"]
        self.num_epochs = conf["epochs"]
        self.batch_size = conf["batch_size"]
        self.verbose = conf["verbose"]
        self.h_act = conf["h_act"]
        self.g_act = conf["g_act"]
        self.corruption_level = conf["corruption_level"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_matrix = dataset.to_csr_matrix()
        self.train_dict = dataset.get_user_train_dict()
        self.sess = sess
        
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None,],name = 'user_input')
            self.input_R = tf.placeholder(tf.float32, [None, self.num_items])
            self.mask_corruption = tf.placeholder(tf.float32, [None, self.num_items])
            
    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            initializer = tool.get_initializer(self.init_method, self.stddev)
            self.V = tf.Variable(initializer([self.num_users, self.hidden_neuron]))
             
            self.weights = {'encoder': tf.Variable(initializer([self.num_items, self.hidden_neuron])),
                            'decoder': tf.Variable(initializer([self.hidden_neuron, self.num_items]))}
            self.biases = {'encoder': tf.Variable(initializer([self.hidden_neuron])),
                           'decoder': tf.Variable(initializer([self.num_items]))}
            
    def _create_inference(self):
        with tf.name_scope("inference"):
            
            self.user_latent = tf.nn.embedding_lookup(self.V, self.user_input)
            
            corrupted_input = tf.multiply(self.input_R, self.mask_corruption)
            encoder_op = tool.activation_function(self.h_act,
                                                  tf.matmul(corrupted_input, self.weights['encoder'])
                                                  + self.biases['encoder'] + self.user_latent)
              
            self.decoder_op = tf.matmul(encoder_op, self.weights['decoder'])+self.biases['decoder']
            self.output = tool.activation_function(self.g_act, self.decoder_op)
            
    def _create_loss(self):
        with tf.name_scope("loss"):
            
            self.loss = - tf.reduce_sum(self.input_R*tf.log(self.output) + (1 - self.input_R)*tf.log(1 - self.output))

            self.reg_loss = self.reg * l2_loss(self.weights['encoder'], self.weights['decoder'],
                                               self.biases['encoder'], self.biases['decoder'],
                                               self.user_latent)
            self.loss = self.loss + self.reg_loss
    
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
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(1, self.num_epochs+1):
            # Generate training instances
            mask_corruption_np = np.random.binomial(1, 1-self.corruption_level, (self.num_users, self.num_items))
            total_loss = 0.0
            training_start_time = time()
            all_users = np.arange(self.num_users)
            users_iter = DataIterator(all_users, batch_size=self.batch_size, shuffle=True, drop_last=False)
            for batch_set_idx in users_iter:
                batch_matrix = np.zeros((len(batch_set_idx), self.num_items))
                for idx, user_id in enumerate(batch_set_idx):
                    items_by_user_id = self.train_dict[user_id]
                    batch_matrix[idx, items_by_user_id] = 1

                feed_dict = {self.mask_corruption: mask_corruption_np[batch_set_idx, :],
                             self.input_R: batch_matrix,
                             self.user_input: batch_set_idx}
                _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                total_loss += loss

            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/self.num_users,
                                                             time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
    
    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_user_ids):
        ratings = []
        mask = np.ones((1, self.num_items), dtype=np.int32)
        if candidate_items_user_ids is not None:
            rating_matrix = np.zeros((1, self.num_items), dtype=np.int32)
            for user_id, candidate_items_user_id in zip(user_ids, candidate_items_user_ids):
                items_by_user_id = self.train_matrix[user_id].indices
                for item_id in items_by_user_id:
                    rating_matrix[0, item_id] = 1
                output = self.sess.run(self.output, 
                                       feed_dict={self.mask_corruption: mask,
                                                  self.input_R: rating_matrix,
                                                  self.user_input: [user_id]})
                ratings.append(output[0, candidate_items_user_id])
                
        else:
            rating_matrix = np.zeros((1,self.num_items), dtype=np.int32)
            all_items = np.arange(self.num_items)
            for user_id in user_ids:
                items_by_user_id = self.train_matrix[user_id].indices
                for item_id in items_by_user_id:
                    rating_matrix[0, item_id] = 1

                feed_dict = {self.mask_corruption: mask,
                             self.input_R: rating_matrix,
                             self.user_input: [user_id]}
                output = self.sess.run(self.output, feed_dict=feed_dict)
                ratings.append(output[0, all_items])
        return ratings
