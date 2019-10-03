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
from neurec.evaluation import Evaluate
from neurec.util.properties import Properties

def csr_to_user_dict(sparse_matrix_data):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    idx_value_dict = {}
    for idx, value in enumerate(sparse_matrix_data):
        if any(value.indices):
            idx_value_dict[idx] = value.indices
    return idx_value_dict


def random_choice(a, size=None, replace=True, p=None, exclusion=None):
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = np.ndarray.flatten(p)
        p[exclusion] = 0
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample


class CFGAN(AbstractRecommender):
    properties = [
        "epochs",
        "topk",
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

    def __init__(self, sess, dataset):
        super().__init__(**kwds)

        self.epochs = self.conf["epochs"]
        self.topK = self.conf["topk"]
        self.mode = self.conf["mode"]
        self.reg_G = self.conf["reg_g"]
        self.reg_D = self.conf["reg_d"]
        self.lr_G = self.conf["lr_g"]
        self.lr_D = self.conf["lr_d"]
        self.batchSize_G = self.conf["batchsize_g"]
        self.batchSize_D = self.conf["batchsize_d"]

        self.opt_G = self.conf["opt_g"]
        self.opt_D = self.conf["opt_d"]
        self.hiddenLayer_G = self.conf["hiddenlayer_g"]
        self.hiddenLayer_D = self.conf["hiddenlayer_d"]
        self.step_G = self.conf["step_g"]
        self.step_D = self.conf["step_d"]

        self.ZR_ratio = self.conf["zr_ratio"]
        self.ZP_ratio = self.conf["zp_ratio"]
        self.ZR_coefficient = self.conf["zr_coefficient"]
        self.verbose= self.conf["verbose"]

        train_matrix = dataset.trainMatrix.tocsr()
        self.train_matrix = train_matrix.copy()
        if self.mode == "itemBased":
            self.train_matrix = self.train_matrix.transpose(copy=True).tocsr()

        self.num_users, self.num_items = self.train_matrix.shape
        self.user_pos_train = csr_to_user_dict(self.train_matrix)
        self.all_items = np.arange(self.num_items)
        self.loss_function = "None"

        # self._build_model()
        # self.sess.run(tf.compat.v1.global_variables_initializer())

    def build_graph(self):
        self._create_layer()

        # generator
        self.condition = tf.compat.v1.placeholder(tf.float32, [None, self.num_items])
        self.G_ZR_dims = tf.compat.v1.placeholder(tf.float32, [None, self.num_items])
        self.G_output = self.gen(self.condition)
        self.G_ZR_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.G_output - 0) * self.G_ZR_dims, 1, keepdims=True))

        # discriminator
        self.mask = tf.compat.v1.placeholder(tf.float32, [None, self.num_items])  # purchased = 1, otherwise 0
        fakeData = self.G_output * self.mask
        fakeData = tf.concat([self.condition, fakeData], 1)

        self.realData = tf.compat.v1.placeholder(tf.float32, [None, self.num_items])
        realData = tf.concat([self.condition, self.realData], 1)

        D_fake = self.dis(fakeData)
        D_real = self.dis(realData)

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')

        # define loss & optimizer for G
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
        g_loss = g_loss + self.reg_G*self._l2loss(g_vars)
        g_loss = g_loss + self.ZR_coefficient*self.G_ZR_loss

        if self.opt_G == 'sgd':
            self.trainer_G = tf.train.GradientDescentOptimizer(self.lr_G).minimize(g_loss, var_list=g_vars)
        elif self.opt_G == 'adam':
            self.trainer_G = tf.train.AdamOptimizer(self.lr_G).minimize(g_loss, var_list=g_vars)

        # define loss & optimizer for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        d_loss = d_loss_real + d_loss_fake + self.reg_D*self._l2loss(d_vars)

        if self.opt_D == 'sgd':
            self.trainer_D = tf.train.GradientDescentOptimizer(self.lr_D).minimize(d_loss, var_list=d_vars)
        elif self.opt_D == 'adam':
            self.trainer_D = tf.train.AdamOptimizer(self.lr_D).minimize(d_loss, var_list=d_vars)

    def _l2loss(self, var):
        l2loss = 0
        for v in var:
            l2loss += tf.nn.l2_loss(v)
        return l2loss

    def _create_layer(self):
        # Generator's layers
        self.gen_layers = []
        xavier_init = tf.contrib.layers.xavier_initializer()
        # stacked hidden layers
        for i, unit in enumerate(self.hiddenLayer_G):
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
        for i, unit in enumerate(self.hiddenLayer_D):
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
            num = int((self.num_items-len(pos_items)) * self.ZR_ratio)
            sample = random_choice(self.all_items, size=num, replace=False, exclusion=pos_items).tolist()
            ZR_matrix[u, sample] = 1

            num = int((self.num_items-len(pos_items)) * self.ZP_ratio)
            sample = random_choice(self.all_items, size=num, replace=False, exclusion=pos_items).tolist()
            PM_matrix[u, sample] = 1
        return train_matrix, csr_matrix(ZR_matrix), csr_matrix(PM_matrix)

    def train_model(self):
        gen_batch_index = np.arange(self.num_users)
        np.random.shuffle(gen_batch_index)
        dis_batch_index = np.arange(self.num_users)
        np.random.shuffle(dis_batch_index)

        totalEpochs = self.epochs
        totalEpochs = int(totalEpochs / self.step_G)
        for epoch in range(totalEpochs):
            train_matrix, ZR_matrix, PM_matrix = self.get_train_data()
            # training discriminator
            for d_epoch in range(self.step_D):
                for idx in np.arange(0, self.num_users, step=self.batchSize_D):
                    idx = dis_batch_index[idx:idx + self.batchSize_D]
                    train_data = train_matrix[idx].toarray()
                    train_mask = PM_matrix[idx].toarray()
                    feed = {self.realData: train_data, self.mask: train_mask, self.condition: train_data}
                    self.sess.run(self.trainer_D, feed_dict=feed)

            # training generator
            for g_epoch in range(self.step_G):
                for idx in np.arange(0, self.num_users, step=self.batchSize_G):
                    idx = dis_batch_index[idx:idx + self.batchSize_G]
                    train_data = train_matrix[idx].toarray()
                    train_z_mask = ZR_matrix[idx].toarray()
                    train_p_mask = PM_matrix[idx].toarray()
                    feed = {self.realData: train_data, self.condition: train_data,
                            self.mask: train_p_mask, self.G_ZR_dims: train_z_mask}
                    self.sess.run(self.trainer_G, feed_dict=feed)
            if epoch %self.verbose == 0:
                self.eval_rating_matrix()
                Evaluate.test_model(self, self.dataset)


    def eval_rating_matrix(self):
        allRatings = self.sess.run(self.G_output, feed_dict={self.condition: self.train_matrix.toarray()})
        if self.mode == "itemBased":
            allRatings = np.transpose(allRatings)
        self.allRatings_for_test = allRatings

    def predict(self, user_id, items):
        return self.allRatings_for_test[user_id, items]
