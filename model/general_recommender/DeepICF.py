"""
Reference: Feng Xue et al., "Deep Item-based Collaborative Filtering for Top-N Recommendation" in TOIS2019
@author: wubin
"""

from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner,data_generator, tool
from util import timer
import pickle
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from util.tool import csr_to_user_dict, pad_sequences
from util import l2_loss
from util.data_iterator import DataIterator


class DeepICF(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(DeepICF, self).__init__(dataset, conf)
        self.pretrain_file = conf["pretrain_file"]
        self.verbose = conf["verbose"]
        self.batch_size = conf["batch_size"]
        self.use_batch_norm = conf["batch_norm"]
        self.num_epochs = conf["epochs"]
        self.weight_size = conf["weight_size"]
        self.embedding_size = conf["embedding_size"]
        self.n_hidden = conf["layers"]
        regs = conf["regs"]
        self.reg_W = conf["regw"]
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.eta_bilinear = regs[2] 
        self.alpha = conf["alpha"]
        self.beta = conf["beta"]
        self.num_negatives = conf["num_neg"]
        self.learning_rate = conf["learning_rate"]
        self.activation = conf["activation"]
        self.algorithm = conf["algorithm"]
        self.learner = conf["learner"]
        self.embed_init_method = conf["embed_init_method"]
        self.weight_init_method = conf["weight_init_method"]
        self.bias_init_method = conf["bias_init_method"]
        self.stddev = conf["stddev"]
        self.dataset = dataset
        self.num_items = dataset.num_items
        self.num_users = dataset.num_users
        self.train_dict = csr_to_user_dict(dataset.train_matrix)
        self.sess = sess

    # batch norm
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])  # the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, ])  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, ])  # the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None, ])  # the ground truth
            self.is_train_phase = tf.placeholder(tf.bool)  # mark is training or testing

    def _create_variables(self, params=None):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            if params is None:
                embed_initializer = tool.get_initializer(self.embed_init_method, self.stddev)
                
                self.c1 = tf.Variable(embed_initializer([self.num_items, self.embedding_size]),
                                      name='c1', dtype=tf.float32)
                self.embedding_Q = tf.Variable(embed_initializer([self.num_items, self.embedding_size]),
                                               name='embedding_Q', dtype=tf.float32)
                self.bias = tf.Variable(tf.zeros(self.num_items), name='bias')
            else:
                self.c1 = tf.Variable(params[0], name='c1', dtype=tf.float32)
                self.embedding_Q = tf.Variable(params[1], name='embedding_Q', dtype=tf.float32)
                self.bias = tf.Variable(params[2], name="bias", dtype=tf.float32)
                
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], axis=0, name='embedding_Q_')
            
            # Variables for attention
            weight_initializer = tool.get_initializer(self.weight_init_method, self.stddev)
            bias_initializer = tool.get_initializer(self.bias_init_method, self.stddev)
            if self.algorithm == 0:
                self.W = tf.Variable(weight_initializer([self.embedding_size, self.weight_size]),
                                     name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            else:
                self.W = tf.Variable(weight_initializer([2 * self.embedding_size, self.weight_size]), 
                                     name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.b = tf.Variable(bias_initializer([1, self.weight_size]), name='Bias_for_MLP',
                                 dtype=tf.float32, trainable=True)
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)

            # Variables for DeepICF+a
            self.weights = {'out': tf.Variable(weight_initializer([self.n_hidden[-1], 1]), name='weights_out')}
            self.biases = {'out': tf.Variable(tf.random_normal([1]), name='biases_out')}
            n_hidden_0 = self.embedding_size
            for i in range(len(self.n_hidden)):
                if i > 0:
                    n_hidden_0 = self.n_hidden[i - 1]
                n_hidden_1 = self.n_hidden[i]
                self.weights['h%d' % i] = tf.Variable(weight_initializer([n_hidden_0, n_hidden_1]),
                                                      name='weights_h%d' % i)
                self.biases['b%d' % i] = tf.Variable(tf.random_normal([n_hidden_1]), name='biases_b%d' % i)

    def _attention_MLP(self, q_):
        with tf.name_scope("attention_MLP"):
            b = tf.shape(q_)[0]
            n = tf.shape(q_)[1]
            r = (self.algorithm + 1) * self.embedding_size

            mlp_output = tf.matmul(tf.reshape(q_, [-1, r]), self.W) + self.b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            if self.activation == 0:
                mlp_output = tf.nn.relu(mlp_output)
            elif self.activation == 1:
                mlp_output = tf.nn.sigmoid(mlp_output)
            elif self.activation == 2:
                mlp_output = tf.nn.tanh(mlp_output)

            A_ = tf.reshape(tf.matmul(mlp_output, self.h), [b, n])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            num_idx = self.num_idx
            mask_mat = tf.sequence_mask(num_idx, maxlen=n, dtype=tf.float32)  # (b, n)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keepdims=True)  # (b, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))

            A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)

            return A, tf.reduce_sum(A * self.embedding_q_, 1)

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, self.user_input)  # (b, n, e)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, self.item_input)  # (b, 1, e)

            if self.algorithm == 0:  # prod
                # (?, k)
                self.A, self.embedding_p = self._attention_MLP(self.embedding_q_ * tf.expand_dims(self.embedding_q, 1))
            else:  # concat
                n = tf.shape(self.user_input)[1]
                self.A, self.embedding_p = self._attention_MLP(tf.concat([self.embedding_q_,
                                                                          tf.tile(tf.expand_dims(self.embedding_q, 1),
                                                                                  tf.stack([1, n, 1]))], 2))  # (?, k)

            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            self.coeff = tf.pow(tf.expand_dims(self.num_idx, 1), tf.constant(self.alpha, tf.float32, [1]))
            self.embedding_p = self.coeff * self.embedding_p  # (?, k)

            # DeepICF+a
            layer1 = tf.multiply(self.embedding_p, self.embedding_q)  # (?, k)
            for i in range(0, len(self.n_hidden)):
                layer1 = tf.add(tf.matmul(layer1, self.weights['h%d' % i]), self.biases['b%d' % i])
                if self.use_batch_norm:
                    layer1 = self.batch_norm_layer(layer1, train_phase=self.is_train_phase, scope_bn='bn_%d' % i)
                layer1 = tf.nn.relu(layer1)
            out_layer = tf.reduce_sum(tf.matmul(layer1, self.weights['out']) + self.biases['out'], 1)  # (?, 1)

            self.output = tf.sigmoid(tf.add_n([out_layer, self.bias_i]))  # (?, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                        self.lambda_bilinear * l2_loss(self.embedding_Q) + \
                        self.gamma_bilinear * l2_loss(self.embedding_Q_) + \
                        self.eta_bilinear * l2_loss(self.W)

            for i in range(min(len(self.n_hidden), len(self.reg_W))):
                if self.reg_W[i] > 0:
                    self.loss = self.loss + self.reg_W[i] * l2_loss(self.weights['h%d' % i])

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
            
    def build_graph(self):
        self._create_placeholders()
        try:
            pretrained_params = []
            with open(self.pretrain_file, "rb") as fin:
                pretrained_params.append(pickle.load(fin, encoding="utf-8"))
            with open(self.mlp_pretrain, "rb") as fin:
                pretrained_params.append(pickle.load(fin, encoding="utf-8"))
            self.logger.info("load pretrained params successful!")
        except:
            pretrained_params = None
            self.logger.info("load pretrained params unsuccessful!")
            
        self._create_variables(pretrained_params)
        self._create_inference()
        self._create_loss()
        self._create_optimizer()

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(1, self.num_epochs+1):
            user_input, num_idx, item_input, labels = \
                data_generator._get_pointwise_all_likefism_data(self.dataset, self.num_negatives, self.train_dict)
            data_iter = DataIterator(user_input, num_idx, item_input, labels,
                                     batch_size=self.batch_size, shuffle=True)
                    
            num_training_instances = len(user_input)
            total_loss = 0.0
            training_start_time = time()
            for bat_users, bat_idx, bat_items, bat_labels in data_iter:
                    bat_users = pad_sequences(bat_users, value=self.num_items)
                    feed_dict = {self.user_input: bat_users,
                                 self.num_idx: bat_idx,
                                 self.item_input: bat_items,
                                 self.labels: bat_labels,
                                 self.is_train_phase: True}
                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/num_training_instances,
                                                             time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
    
    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)
    
    def predict(self, user_ids, candidate_items_userids):      
        ratings = []
        if candidate_items_userids is not None:
            for u, eval_items_by_u in zip(user_ids, candidate_items_userids):
                user_input = []
                cand_items_by_u = self.train_dict[u]
                num_idx = len(cand_items_by_u)
                item_idx = np.full(len(eval_items_by_u), num_idx, dtype=np.int32)
                user_input.extend([cand_items_by_u]*len(eval_items_by_u))
                feed_dict = {self.user_input: user_input,
                             self.num_idx: item_idx, 
                             self.item_input: eval_items_by_u,
                             self.is_train_phase: False}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
                
        else:
            eval_items = np.arange(self.num_items)
            for u in user_ids:
                user_input = []
                cand_items_by_u = self.train_dict[u]
                num_idx = len(cand_items_by_u)
                item_idx = np.full(self.num_items, num_idx, dtype=np.int32)
                user_input.extend([cand_items_by_u]*self.num_items)
                feed_dict = {self.user_input: user_input,
                             self.num_idx: item_idx, 
                             self.item_input: eval_items,
                             self.is_train_phase: False}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        return ratings
