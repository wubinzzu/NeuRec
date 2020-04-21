"""
Xiangnan He et al., "Outer Product-based Neural Collaborative Filtering", In IJCAI 2018.  
@author: wubin
"""
from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool
from util import timer
import pickle
from util import l2_loss
from data import PairwiseSampler


class ConvNCF(AbstractRecommender):
    def __init__(self, sess, dataset, conf):  
        super(ConvNCF, self).__init__(dataset, conf)
        self.embedding_size = conf["embedding_size"]
        regs = conf["regs"]
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.lambda_weight = regs[2]
        self.keep = conf["keep"]
        self.num_epochs = conf["epochs"]
        self.batch_size = conf["batch_size"]
        self.nc = conf["net_channel"]
        self.lr_embed = conf["lr_embed"]
        self.lr_net = conf["lr_net"]
        self.verbose = conf["verbose"]
        self.loss_function = conf["loss_function"]
        self.num_negatives = conf["num_negatives"]
        self.embed_init_method = conf["embed_init_method"]
        self.weight_init_method = conf["weight_init_method"]
        self.stddev = conf["stddev"]
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items 
        self.dataset = dataset
        self.sess = sess
        
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None,], name = "user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape = [None,], name = "item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape = [None,], name = "item_input_neg")
            self.keep_prob = tf.placeholder_with_default(1.0, shape=None,name = "keep_prob")

    # ---------- model definition -------
    def weight_variable(self, shape):
        initializer = tool.get_initializer(self.weight_init_method, self.stddev)
        return tf.Variable(initializer(shape))

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv_weight(self, isz, osz):
        return self.weight_variable([2,2,isz,osz]), self.bias_variable([osz])

    def _conv_layer(self, x, P):
        conv = tf.nn.conv2d(x, P[0], strides=[1, 2, 2, 1], padding='SAME')
        return tf.nn.tanh(conv + P[1])

    def _create_variables(self, params=None):
        with tf.name_scope("embedding"):
            if params is None:
                initializer = tool.get_initializer(self.embed_init_method, self.stddev)
                self.embedding_P = tf.Variable(initializer([self.num_users, self.embedding_size]),
                                               name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
                self.embedding_Q = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                               name='embedding_Q', dtype=tf.float32)  # (items, embedding_size)
            
            else:
                self.embedding_P = tf.Variable(params[0], name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
                self.embedding_Q = tf.Variable(params[1], name='embedding_Q', dtype=tf.float32)  # (items, embedding_size)

            # here should have 6 iszs due to the size of outer products is 64x64
            iszs = [1] + self.nc[:-1]
            oszs = self.nc
            self.P = []
            for isz, osz in zip(iszs, oszs):
                self.P.append(self._conv_weight(isz, osz))

            self.W = self.weight_variable([self.nc[-1], 1])  # 32x1
            self.b = self.weight_variable([1])  # 1

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            self.embedding_p = tf.nn.embedding_lookup(self.embedding_P, self.user_input)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, item_input)

            # outer product of P_u and Q_i
            self.relation = tf.matmul(tf.expand_dims(self.embedding_p,2), tf.expand_dims(self.embedding_q,1))
            self.net_input = tf.expand_dims(self.relation, -1)

            # CNN
            self.layer = []
            input = self.net_input
            for p in self.P:
                self.layer.append(self._conv_layer(input, p))
                input = self.layer[-1]
            # prediction
            self.dropout = tf.nn.dropout(self.layer[-1], self.keep_prob)
            self.output_layer = tf.matmul(tf.reshape(self.dropout,[-1,self.nc[-1]]), self.W) + self.b

            return self.embedding_p, self.embedding_q, self.output_layer

    def _regular(self, params):
        res = 0
        for param in params:
            res += l2_loss(param[0], param[1])
        return res

    def _create_loss(self):
        with tf.name_scope("loss"):
            # BPR loss for L(Theta)
            self.p1, self.q1, self.output = self._create_inference(self.item_input_pos)
            self.p2, self.q2, self.output_neg = self._create_inference(self.item_input_neg)
            self.result = self.output - self.output_neg
            self.loss = learner.pairwise_loss(self.loss_function, self.result)

            self.opt_loss = self.loss + self.lambda_bilinear * l2_loss(self.p1, self.q2, self.q1) + \
                            self.gamma_bilinear * self._regular([(self.W, self.b)]) + \
                            self.lambda_weight * (self._regular(self.P) + self._regular([(self.W, self.b)]))

    # used at the first time when emgeddings are pretrained yet network are randomly initialized
    # if not, the parameters may be NaN.
    def _create_pre_optimizer(self):
        self.pre_opt = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.loss)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            # seperated optimizer
            var_list1 = [self.embedding_P, self.embedding_Q]
            # [self.W1,self.W2,self.W3,self.W4,self.b1,self.b2,self.b3,self.b4,self.P1,self.P2,self.P3]
            var_list2 = list(set(tf.trainable_variables()) - set(var_list1))
            opt1 = tf.train.AdagradOptimizer(self.lr_embed)
            opt2 = tf.train.AdagradOptimizer(self.lr_net)
            grads = tf.gradients(self.opt_loss, var_list1 + var_list2)
            grads1 = grads[:len(var_list1)]
            grads2 = grads[len(var_list1):]
            train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
            train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
            self.optimizer = tf.group(train_op1, train_op2)   
            
    def build_graph(self):
        self._create_placeholders()
        try:
            pretrained_params = []
            with open(self.mf_pretrain, "rb") as fin:
                pretrained_params.append(pickle.load(fin, encoding="utf-8"))
            with open(self.mlp_pretrain, "rb") as fin:
                pretrained_params.append(pickle.load(fin, encoding="utf-8"))
            self.logger.info("load pretrained params successful!")
        except:
            pretrained_params = None
            self.logger.info("load pretrained params unsuccessful!")
        self._create_variables(pretrained_params)
        self._create_loss()
        if pretrained_params is None:
            self._create_pre_optimizer()
        self._create_optimizer()

    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        data_iter = PairwiseSampler(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.num_epochs+1):
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(data_iter)
            for bat_users, bat_items_pos, bat_items_neg in data_iter:
                    feed_dict = {self.user_input: bat_users,
                                 self.item_input_pos: bat_items_pos,
                                 self.item_input_neg: bat_items_neg}
                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss / num_training_instances,
                                                             time() - training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_user_ids):
        ratings = []
        if candidate_items_user_ids is not None:
            for u, i in zip(user_ids, candidate_items_user_ids):
                users = np.full(len(i), u, dtype=np.int32)
                feed_dict = {self.user_input: users, self.item_input_pos: i}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        else:
            for u in user_ids:
                users = np.full(self.num_items, u, dtype=np.int32)
                feed_dict = {self.user_input: users, self.item_input_pos: np.arange(self.num_items)}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        return ratings
