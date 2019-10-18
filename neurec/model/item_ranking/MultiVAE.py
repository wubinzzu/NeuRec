'''
Reference: Dawen, Liang, et al. "Variational autoencoders for collaborative filtering." in WWW2018
@author: wubin
'''
import tensorflow as tf
import numpy as np
from time import time
from neurec.util import learner, tool
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
from neurec.evaluation import Evaluate
from neurec.model.AbstractRecommender import AbstractRecommender
from neurec.util.properties import Properties

class MultiVAE(AbstractRecommender):
    properties = [
        "learning_rate",
        "learner",
        "topk",
        "batch_size",
        "p_dim",
        "activation",
        "reg",
        "epochs",
        "verbose",
        "anneal_cap",
        "total_anneal_steps"
    ]

    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.learning_rate = self.conf["learning_rate"]
        self.learner = self.conf["learner"]
        self.topK = self.conf["topk"]
        self.batch_size= self.conf["batch_size"]
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.p_dims = self.conf["p_dim"] + [self.num_items]
        self.q_dims = self.p_dims[::-1]
        self.dims = self.q_dims + self.p_dims[1:]
        self.act = self.conf["activation"]
        self.reg = self.conf["reg"]
        self.num_epochs=self.conf["epochs"]
        self.loss_function="multinominal-likelihood"
        self.verbose=self.conf["verbose"]
        self.anneal_cap=self.conf["anneal_cap"]
        self.total_anneal_steps=self.conf["total_anneal_steps"]

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
            self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)
            self.is_training_ph = tf.placeholder_with_default(0., shape=None)
            self.anneal_ph = tf.placeholder_with_default(1., shape=None)

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            self.weights_q, self.biases_q = [], []

            for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
                if i == len(self.q_dims[:-1]) - 1:
                    # we need two sets of parameters for mean and variance,
                    # respectively
                    d_out *= 2
                weight_key = "weight_q_{}to{}".format(i, i+1)
                bias_key = "bias_q_{}".format(i+1)

                self.weights_q.append(tf.get_variable(
                    name=weight_key, shape=[d_in, d_out],
                    initializer=tf.contrib.layers.xavier_initializer()))

                self.biases_q.append(tf.get_variable(
                    name=bias_key, shape=[d_out],
                    initializer=tf.truncated_normal_initializer(
                        stddev=0.001)))


            self.weights_p, self.biases_p = [], []

            for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
                weight_key = "weight_p_{}to{}".format(i, i+1)
                bias_key = "bias_p_{}".format(i+1)
                self.weights_p.append(tf.get_variable(
                    name=weight_key, shape=[d_in, d_out],
                    initializer=tf.contrib.layers.xavier_initializer()))

                self.biases_p.append(tf.get_variable(
                    name=bias_key, shape=[d_out],
                    initializer=tf.truncated_normal_initializer(
                        stddev=0.001)))
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
                logvar_q = h[:, self.q_dims[-1]:]#log sigmod^2  batch x 200

                std_q = tf.exp(0.5 * logvar_q) #sigmod batch x 200
                KL = tf.reduce_mean(tf.reduce_sum(
                        0.5 * (-logvar_q + tf.exp(logvar_q) + tf.pow(mu_q,2) - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        self.h = z

        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            self.h = tf.matmul(self.h, w) + b

            if i != len(self.weights_p) - 1:
                self.h = tool.activation_function(self.act,self.h)
        return self.h

    def _create_inference(self):
        with tf.name_scope("inference"):
            # q-network
            mu_q, std_q, self.KL = self.q_graph()
            epsilon = tf.random_normal(tf.shape(std_q))

            sampled_z = mu_q + self.is_training_ph *\
                epsilon * std_q # batch x 200

            # p-network
            logits = self.p_graph(sampled_z)

            self.log_softmax_var = tf.nn.log_softmax(logits)


    def _create_loss(self):
        with tf.name_scope("loss"):
            neg_ll = -tf.reduce_mean(tf.reduce_sum(
            self.log_softmax_var * self.input_ph,
            axis=1))
            # apply regularization to weights
            reg = l2_regularizer(self.reg)

            reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
            # tensorflow l2 regularization multiply 0.5 to the l2 norm
            # multiply 2 so that it is back in the same scale
            self.loss = neg_ll + self.anneal_ph * self.KL + 2 * reg_var #neg_ELBO

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
        # the total number of gradient updates for annealing
        # largest annealing parameter
        for epoch in  range(self.num_epochs):
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

                batch_matrix = np.zeros((len(batch_set_idx),self.num_items))

                if self.total_anneal_steps > 0:
                    anneal = min(self.anneal_cap, 1. * update_count / self.total_anneal_steps)
                else:
                    anneal = self.anneal_cap

                batch_uid = 0
                trainDict = self.dataset.trainDict
                for userid in batch_set_idx:
                    items_by_userid = trainDict[userid]
                    for itemid in items_by_userid:
                        batch_matrix[batch_uid,itemid] = 1

                    batch_uid=batch_uid+1

                feed_dict = feed_dict={self.input_ph: batch_matrix,self.keep_prob_ph: 0.5,
                            self.anneal_ph: anneal,self.is_training_ph: 1}
                _, loss = self.sess.run([self.optimizer, self.loss],feed_dict=feed_dict)
                total_loss+=loss

                update_count += 1
            self.logger.info("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)

    def predict(self, user_id, items):
        rating_matrix = np.zeros((1,self.num_items), dtype=np.int32)
        items_by_userid = self.dataset.trainDict[user_id]
        for itemid in items_by_userid:
            rating_matrix[0,itemid] = 1
        output = self.sess.run(self.h, feed_dict={self.input_ph:rating_matrix})
        return output[0,items]
