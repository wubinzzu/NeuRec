"""
Reference: Ziwei Zhu, et al. "Improving Top-K Recommendation via Joint
Collaborative Autoencoders." in WWW2019
@author: wubin
"""
from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool
from util import timer
from util import l2_loss


class JCA(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(JCA, self).__init__(dataset, conf)
        self.hidden_neuron = conf["hidden_neuron"]
        self.learning_rate = conf["learning_rate"]
        self.learner = conf["learner"]
        self.reg = conf["reg"]
        self.num_epochs = conf["epochs"]
        self.batch_size = conf["batch_size"]
        self.verbose = conf["verbose"]
        self.f_act = conf["f_act"]  # the activation function for the output layer
        self.g_act = conf["g_act"]  # the activation function for the hidden layer
        self.margin = conf["margin"]
        self.corruption_level = conf["corruption_level"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.neg_sample_rate = conf["num_neg"]
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items 
        self.dataset = dataset
        self.train_R = dataset.train_matrix.toarray()
        self.U_OH_mat = np.eye(self.num_users, dtype=float)
        self.I_OH_mat = np.eye(self.num_items, dtype=float)
        self.num_batch_U = int(self.num_users / float(self.batch_size)) + 1
        self.num_batch_I = int(self.num_items / float(self.batch_size)) + 1
        self.sess = sess
        
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            # input rating vector
            self.input_R_U = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R_U")
            self.input_R_I = tf.placeholder(dtype=tf.float32, shape=[self.num_users, None], name="input_R_I")
            self.input_OH_I = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_OH_I")
            self.input_P_cor = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="input_P_cor")
            self.input_N_cor = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="input_N_cor")
    
            # input indicator vector indicator
            self.row_idx = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="row_idx")
            self.col_idx = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="col_idx")
            
    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            initializer = tool.get_initializer(self.init_method, self.stddev)
            # user component
            # first layer weights
            self.UV = tf.Variable(initializer([self.num_items, self.hidden_neuron]), name="UV", dtype=tf.float32)
            # second layer weights
            self.UW = tf.Variable(initializer([self.hidden_neuron, self.num_items]), name="UW", dtype=tf.float32)
            # first layer bias
            self.Ub1 = tf.Variable(initializer([1, self.hidden_neuron]), name="Ub1", dtype=tf.float32)
            # second layer bias
            self.Ub2 = tf.Variable(initializer([1, self.num_items]), name="Ub2", dtype=tf.float32)
    
            # item component
            # first layer weights
            self.IV = tf.Variable(initializer([self.num_users, self.hidden_neuron]), name="IV", dtype=tf.float32)
            # second layer weights
            self.IW = tf.Variable(initializer([self.hidden_neuron, self.num_users]), name="IW", dtype=tf.float32)
            # first layer bias
            self.Ib1 = tf.Variable(initializer([1, self.hidden_neuron]), name="Ib1", dtype=tf.float32)
            # second layer bias
            self.Ib2 = tf.Variable(initializer([1, self.num_users]), name="Ib2", dtype=tf.float32)

            self.I_factor_vector = tf.Variable(initializer([1, self.num_items]),name="I_factor_vector", dtype=tf.float32)

    def _create_inference(self):
        with tf.name_scope("inference"):
            
            # user component
            u_pre_encoder = tf.matmul(self.input_R_U, self.UV) + self.Ub1  # input to the hidden layer
            self.u_encoder = tool.activation_function(self.g_act, u_pre_encoder)  # output of the hidden layer
            u_pre_decoder = tf.matmul(self.u_encoder, self.UW) + self.Ub2  # input to the output layer
            self.u_decoder = tool.activation_function(self.f_act, u_pre_decoder)  # output of the output layer
    
            # item component
            i_pre_mul = tf.transpose(tf.matmul(self.I_factor_vector, tf.transpose(self.input_OH_I)))
            i_pre_encoder = tf.matmul(tf.transpose(self.input_R_I), self.IV) + self.Ib1  # input to the hidden layer
            self.I_Encoder = tool.activation_function(self.g_act, i_pre_encoder * i_pre_mul)  # output of the hidden layer
            i_pre_decoder = tf.matmul(self.I_Encoder, self.IW) + self.Ib2  # input to the output layer
            self.I_Decoder = tool.activation_function(self.f_act, i_pre_decoder)  # output of the output layer
    
            # final output
            self.Decoder = ((tf.transpose(tf.gather_nd(tf.transpose(self.u_decoder), self.col_idx)))
                            + tf.gather_nd(tf.transpose(self.I_Decoder), self.row_idx)) / 2.0
    
            pos_data = tf.gather_nd(self.Decoder, self.input_P_cor)
            neg_data = tf.gather_nd(self.Decoder, self.input_N_cor)
            
            self.pre_cost1 = tf.maximum(neg_data - pos_data + self.margin, tf.zeros(tf.shape(neg_data)[0]))
            
    def _create_loss(self):
        with tf.name_scope("loss"):
            
            cost1 = tf.reduce_sum(self.pre_cost1)  # prediction squared error
            pre_cost2 = l2_loss(self.UW, self.UV, self.IW, self.IV, self.Ib1, self.Ib2, self.Ub1, self.Ub2)
            cost2 = self.reg * 0.5 * pre_cost2  # regularization term
    
            self.cost = cost1 + cost2  # the loss function

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.cost, self.learning_rate)
            
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
                                               
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in  range(1,self.num_epochs+1):
            random_row_idx = np.random.permutation(self.num_users)  # randomly permute the rows
            random_col_idx = np.random.permutation(self.num_items)  # randomly permute the cols
            training_start_time = time()
            total_loss = 0.0
            for i in range(self.num_batch_U):  # iterate each batch
                if i == self.num_batch_U - 1:
                    row_idx = random_row_idx[i * self.batch_size:]
                else:
                    row_idx = random_row_idx[(i * self.batch_size):((i + 1) * self.batch_size)]
                for j in range(self.num_batch_I):
                    # get the indices of the current batch
                    if j == self.num_batch_I - 1:
                        col_idx = random_col_idx[j * self.batch_size:]
                    else:
                        col_idx = random_col_idx[(j * self.batch_size):((j + 1) * self.batch_size)]
                    
                    p_input, n_input = self.pairwise_neg_sampling(row_idx, col_idx)
                    input_tmp = self.train_R[row_idx, :]
                    input_tmp = input_tmp[:, col_idx]
    
                    input_r_u = self.train_R[row_idx, :]
                    input_r_i = self.train_R[:, col_idx]
                    _, loss = self.sess.run(  # do the optimization by the minibatch
                        [self.optimizer, self.cost],
                        feed_dict={
                            self.input_R_U: input_r_u,
                            self.input_R_I: input_r_i,
                            self.input_OH_I: self.I_OH_mat[col_idx, :],
                            self.input_P_cor: p_input,
                            self.input_N_cor: n_input,
                            self.row_idx: np.reshape(row_idx, (len(row_idx), 1)),
                            self.col_idx: np.reshape(col_idx, (len(col_idx), 1))})
                total_loss += loss
            self.logger.info("[iter %d : loss : %f, time: %f]" %
                             (epoch, total_loss, time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
    
    @timer
    def evaluate(self):
        feed_dict = {self.input_R_U: self.train_R,
                     self.input_R_I: self.train_R,
                     self.input_OH_I: self.I_OH_mat,
                     self.input_P_cor: [[0, 0]],
                     self.input_N_cor: [[0, 0]],
                     self.row_idx: np.reshape(range(self.num_users), (self.num_users, 1)),
                     self.col_idx: np.reshape(range(self.num_items), (self.num_items, 1))}

        self.all_ratings = self.sess.run(self.Decoder, feed_dict=feed_dict)
        return self.evaluator.evaluate(self)
                
    def pairwise_neg_sampling(self,row_idx, col_idx):
        R = self.train_R[row_idx, :]
        R = R[:, col_idx]
        p_input, n_input = [], []
        obsv_list = np.where(R == 1)
    
        unobsv_mat = []
        for r in range(R.shape[0]):
            unobsv_list = np.where(R[r, :] == 0)
            unobsv_list = unobsv_list[0]
            unobsv_mat.append(unobsv_list)
    
        for i in range(len(obsv_list[1])):
            # positive instance
            u = obsv_list[0][i]
            # negative instances
            unobsv_list = unobsv_mat[u]
            neg_sample_list = np.random.choice(unobsv_list, size=self.neg_sample_rate, replace=False)
            for ns in neg_sample_list:
                p_input.append([u, obsv_list[1][i]])
                n_input.append([u, ns])
        return np.array(p_input), np.array(n_input)

    def predict(self, user_ids, candidate_items_user_ids):
        ratings = []
        if candidate_items_user_ids is None:
            all_items = np.arange(self.num_items)
            for user_id in user_ids:
                ratings.append(self.all_ratings[user_id, all_items])
             
        else:
            for user_id, candidate_items_user_id in zip(user_ids, candidate_items_user_ids):
                ratings.append(self.all_ratings[user_id, candidate_items_user_id])
             
        return ratings
