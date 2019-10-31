'''
Reference: Dong-Kyu Chae, et al., "CFGAN: A Generic Collaborative Filtering Framework 
based on Generative Adversarial Networks." in CIKM2018
@author: Zhongchuan Sun
'''
# ZP: Hybrid of zero-reconstruction regularization and partial-masking

from neurec.model.AbstractRecommender import AbstractRecommender
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from neurec.util.tool import csr_to_user_dict
from neurec.util.tool import randint_choice
from neurec.util.tool import l2_loss
from neurec.util.DataIterator import DataIterator

class CFGAN(AbstractRecommender):
    properties = [
        "epochs",
        "mode",
        "reg_g",
        "reg_d",
        "lr_g",
        "lr_d",
        "batchsize_g",
        "batchsize_d",
        "opt_g",
        "opt_d",
        "hiddenlayer_g",
        "hiddenlayer_d",
        "step_g",
        "step_d",
        "zr_ratio",
        "zp_ratio",
        "zr_coefficient",
        "verbose"
    ]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        
        self.epochs = self.conf["epochs"]
        self.mode = self.conf["mode"]
        self.reg_g = self.conf["reg_g"]
        self.reg_d = self.conf["reg_d"]
        self.lr_g = self.conf["lr_g"]
        self.lr_d = self.conf["lr_d"]
        self.batchsize_g = self.conf["batchsize_g"]
        self.batchsize_d = self.conf["batchsize_d"]

        self.opt_g = self.conf["opt_g"]
        self.opt_d = self.conf["opt_d"]
        self.hiddenlayer_g = self.conf["hiddenlayer_g"]
        self.hiddenlayer_d = self.conf["hiddenlayer_d"]
        self.step_g = self.conf["step_g"]
        self.step_d = self.conf["step_d"]

        self.zr_ratio = self.conf["zr_ratio"]
        self.zp_ratio = self.conf["zp_ratio"]
        self.zr_coefficient = self.conf["zr_coefficient"]
        self.verbose = self.conf["verbose"]
        
        train_matrix = self.dataset.train_matrix
        self.train_matrix = train_matrix.copy()
        if self.mode == "itemBased":
            self.train_matrix = self.train_matrix.transpose(copy=True).tocsr()

        self.num_users, self.num_items = self.train_matrix.shape
        self.user_pos_train = csr_to_user_dict(self.train_matrix)
        self.all_items = np.arange(self.num_items)
        

    def build_graph(self):
        self._create_layer()

        # generator
        self.condition = tf.placeholder(tf.float32, [None, self.num_items])
        self.G_ZR_dims = tf.placeholder(tf.float32, [None, self.num_items])
        self.G_output = self.gen(self.condition)
        self.G_ZR_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.G_output - 0) * self.G_ZR_dims, 1, keepdims=True))

        # discriminator
        self.mask = tf.placeholder(tf.float32, [None, self.num_items])  # purchased = 1, otherwise 0
        fakeData = self.G_output * self.mask
        fakeData = tf.concat([self.condition, fakeData], 1)

        self.realData = tf.placeholder(tf.float32, [None, self.num_items])
        realData = tf.concat([self.condition, self.realData], 1)

        D_fake = self.dis(fakeData)
        D_real = self.dis(realData)

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')

        # define loss & optimizer for G
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

        g_loss = g_loss + self.reg_g * l2_loss(*g_vars)
        g_loss = g_loss + self.zr_coefficient*self.G_ZR_loss

        if self.opt_g == 'sgd':
            self.trainer_G = tf.train.GradientDescentOptimizer(self.lr_g).minimize(g_loss, var_list=g_vars)
        elif self.opt_g == 'adam':
            self.trainer_G = tf.train.AdamOptimizer(self.lr_g).minimize(g_loss, var_list=g_vars)

        # define loss & optimizer for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        d_loss = d_loss_real + d_loss_fake + self.reg_d * l2_loss(*d_vars)

        if self.opt_d == 'sgd':
            self.trainer_D = tf.train.GradientDescentOptimizer(self.lr_d).minimize(d_loss, var_list=d_vars)
        elif self.opt_d == 'adam':
            self.trainer_D = tf.train.AdamOptimizer(self.lr_d).minimize(d_loss, var_list=d_vars)

    def _create_layer(self):
        # Generator's layers
        self.gen_layers = []
        xavier_init = tf.contrib.layers.xavier_initializer()
        # stacked hidden layers
        for i, unit in enumerate(self.hiddenlayer_g):
            hidden_layer = tf.layers.Dense(unit, activation=tf.sigmoid,
                                           kernel_initializer=xavier_init, name="gen_h%d"%i)
            self.gen_layers.append(hidden_layer)

        # hidden -> output
        output_layer = tf.layers.Dense(self.num_items, activation=tf.identity,
                                       kernel_initializer=xavier_init, name="gen_out")
        self.gen_layers.append(output_layer)

        # Discriminator's layers
        self.dis_layers = []
        # stacked hidden layers
        for i, unit in enumerate(self.hiddenlayer_d):
            hidden_layer = tf.layers.Dense(unit, activation=tf.sigmoid,
                                           kernel_initializer=xavier_init, name="dis_h%d"%i)
            self.dis_layers.append(hidden_layer)

        # hidden -> output
        output_layer = tf.layers.Dense(1, activation=tf.identity,
                                       kernel_initializer=xavier_init, name="dis_out")
        self.dis_layers.append(output_layer)

    def gen(self, input):
        for layer in self.gen_layers:
            input = layer.apply(input)
        return input

    def dis(self, input):
        for layer in self.dis_layers:
            input = layer.apply(input)
        return input

    def get_train_data(self):
        train_matrix = self.train_matrix.copy()
        ZR_matrix = self.train_matrix.todense()
        PM_matrix = self.train_matrix.todense()

        for u, pos_items in self.user_pos_train.items():
            num = int((self.num_items-len(pos_items)) * self.zr_ratio)
            sample = randint_choice(self.num_items, size=num, replace=False, exclusion=pos_items).tolist()
            ZR_matrix[u, sample] = 1

            num = int((self.num_items-len(pos_items)) * self.zp_ratio)
            sample = randint_choice(self.num_items, size=num, replace=False, exclusion=pos_items).tolist()
            PM_matrix[u, sample] = 1
        return train_matrix, csr_matrix(ZR_matrix), csr_matrix(PM_matrix)

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        G_iter = DataIterator(np.arange(self.num_users), batch_size=self.batchsize_g, shuffle=True, drop_last=False)
        D_iter = DataIterator(np.arange(self.num_users), batch_size=self.batchsize_d, shuffle=True, drop_last=False)

        totalEpochs = self.epochs
        totalEpochs = int(totalEpochs / self.step_g)
        for epoch in range(totalEpochs):
            train_matrix, ZR_matrix, PM_matrix = self.get_train_data()
            # training discriminator
            for d_epoch in range(self.step_d):
                for idx in D_iter:
                    train_data = train_matrix[idx].toarray()
                    train_mask = PM_matrix[idx].toarray()
                    feed = {self.realData: train_data, self.mask: train_mask, self.condition: train_data}
                    self.sess.run(self.trainer_D, feed_dict=feed)

            # training generator
            for g_epoch in range(self.step_g):
                for idx in G_iter:
                    train_data = train_matrix[idx].toarray()
                    train_z_mask = ZR_matrix[idx].toarray()
                    train_p_mask = PM_matrix[idx].toarray()
                    feed = {self.realData: train_data, self.condition: train_data,
                            self.mask: train_p_mask, self.G_ZR_dims: train_z_mask}
                    self.sess.run(self.trainer_G, feed_dict=feed)
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    def evaluate(self):
        self.eval_rating_matrix()
        return self.evaluator.evaluate(self)

    def eval_rating_matrix(self):
        allRatings = self.sess.run(self.G_output, feed_dict={self.condition: self.train_matrix.toarray()})
        if self.mode == "itemBased":
            allRatings = np.transpose(allRatings)
        self.allRatings_for_test = allRatings

    def predict(self, users, items):
        all_ratings = self.allRatings_for_test[users]
        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]
        return all_ratings
