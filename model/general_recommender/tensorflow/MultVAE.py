"""
Paper: Variational Autoencoders for Collaborative Filtering
Author: Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara
"""

__author__ = "Bin Wu"
__email__ = ""

__all__ = ["MultVAE"]

import tensorflow as tf
from model.base import AbstractRecommender
from util.tensorflow import l2_loss
from util.tensorflow import get_initializer, get_session
import numpy as np
from reckit import DataIterator


class MultVAE(AbstractRecommender):
    def __init__(self, config):
        super(MultVAE, self).__init__(config)
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.keep_prob = config["keep_prob"]
        self.total_anneal_steps = config["total_anneal_steps"]
        self.anneal_cap = config["anneal_cap"]

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.param_init = config["param_init"]

        self.train_csr_mat = self.dataset.train_data.to_csr_matrix()
        self.train_csr_mat.data[:] = 1.0
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        p_dims = config["p_dims"]
        self.p_dims = p_dims + [self.num_items]
        if ("q_dims" not in config) or (config["q_dims"] is None):
            self.q_dims = self.p_dims[::-1]
        else:
            q_dims = config["q_dims"]
            q_dims = [self.num_items] + q_dims
            assert q_dims[0] == self.p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == self.p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims

        self._build_model()
        self.sess = get_session(config["gpu_mem"])

    def _create_variable(self):
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)

        # embedding layers
        init = get_initializer(self.param_init)

        self.weights_q, self.biases_q = [], []
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance, respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            bias_key = "bias_q_{}".format(i + 1)
            self.weights_q.append(tf.Variable(init([d_in, d_out]), name=weight_key))
            self.biases_q.append(tf.Variable(init([d_out]), name=bias_key))

        self.weights_p, self.biases_p = [], []
        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i + 1)
            bias_key = "bias_p_{}".format(i + 1)
            self.weights_p.append(tf.Variable(init([d_in, d_out]), name=weight_key))
            self.biases_p.append(tf.Variable(init([d_out]), name=bias_key))

    def q_graph(self):
        mu_q, std_q, kl_dist = None, None, None

        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q, logvar_q = tf.split(h, 2, axis=1)
                std_q = tf.exp(0.5 * logvar_q)
                kl_dist = tf.reduce_mean(tf.reduce_sum(0.5*(-logvar_q+tf.exp(logvar_q) + mu_q**2 - 1), axis=1))
        return mu_q, std_q, kl_dist

    def p_graph(self, z):
        h = z

        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, kl_dist = self.q_graph()
        epsilon = tf.random_normal(tf.shape(std_q))

        sampled_z = mu_q + self.is_training_ph*epsilon*std_q

        # p-network
        logits = self.p_graph(sampled_z)

        return logits, kl_dist

    def _build_model(self):
        self._create_variable()
        logits, kl_dist = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits, axis=-1)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var*self.input_ph, axis=-1))
        # apply regularization to weights
        reg_var = l2_loss(*self.weights_q) + l2_loss(*self.weights_p)
        reg_var *= self.reg

        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_elbo = neg_ll + self.anneal_ph*kl_dist + 2*reg_var

        self.train_opt = tf.train.AdamOptimizer(self.lr).minimize(neg_elbo)

        # for evaluation
        self.batch_ratings = logits

    def train_model(self):
        train_users = [user for user in range(self.num_users) if self.train_csr_mat[user].nnz]
        user_iter = DataIterator(train_users, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())

        update_count = 0.0
        for epoch in range(self.epochs):
            for bat_users in user_iter:
                bat_input = self.train_csr_mat[bat_users]

                if self.total_anneal_steps > 0:
                    anneal = min(self.anneal_cap, 1.*update_count/self.total_anneal_steps)
                else:
                    anneal = self.anneal_cap

                feed_dict = {self.input_ph: bat_input.toarray(),
                             self.keep_prob_ph: self.keep_prob,
                             self.anneal_ph: anneal,
                             self.is_training_ph: 1}
                self.sess.run(self.train_opt, feed_dict=feed_dict)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users = np.asarray(users)
        sp_mat = self.train_csr_mat[users]

        feed_dict = {self.input_ph: sp_mat.toarray()}
        all_ratings = self.sess.run(self.batch_ratings, feed_dict=feed_dict)
        return all_ratings
