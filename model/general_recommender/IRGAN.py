"""
Reference: Jun Wang, et al., "IRGAN: A Minimax Game for Unifying Generative and 
Discriminative Information Retrieval Models." SIGIR 2017.
@author: Zhongchuan Sun
"""
from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from util import l2_loss
from util.data_iterator import DataIterator


class GEN(object):
    def __init__(self, item_num, user_num, emb_dim, lamda, param=None, init_delta=0.05, learning_rate=0.05):
        self.itemNum = item_num
        self.userNum = user_num
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.init_delta = init_delta
        self.learning_rate = learning_rate
        self.g_params = []

        with tf.variable_scope('generator'):
            if self.param is None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.init_delta, maxval=self.init_delta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.init_delta, maxval=self.init_delta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(param[2])

            self.g_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.i_prob = tf.gather(
            tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]),
            self.i)

        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) + \
                        self.lamda * l2_loss(self.u_embedding, self.i_embedding, self.i_bias)

        g_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.gan_updates = g_opt.minimize(self.gan_loss, var_list=self.g_params)

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias


class DIS(object):
    def __init__(self, item_num, user_num, emb_dim, lamda, param=None, init_delta=0.05, learning_rate=0.05):
        self.itemNum = item_num
        self.userNum = user_num
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.init_delta = init_delta
        self.learning_rate = learning_rate
        self.d_params = []

        with tf.variable_scope('discriminator'):
            if self.param is None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.init_delta, maxval=self.init_delta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.init_delta, maxval=self.init_delta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(self.param[2])

        self.d_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        self.pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.pre_logits) + \
                        self.lamda * l2_loss(self.u_embedding, self.i_embedding, self.i_bias)

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)

        self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),
                                           1) + self.i_bias
        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_b=True) + self.item_bias

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.NLL = -tf.reduce_mean(tf.log(
            tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))
        )
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias


class IRGAN(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(IRGAN, self).__init__(dataset, conf)

        train_matrix = dataset.trainMatrix.tocsr()
        self.num_users, self.num_items = train_matrix.shape
        self.factors_num = conf["factors_num"]
        self.lr = conf["lr"]
        self.g_reg = conf["g_reg"]
        self.d_reg = conf["d_reg"]
        self.epochs = conf["epochs"]
        self.g_epoch = conf["g_epoch"]
        self.d_epoch = conf["d_epoch"]
        self.batch_size = conf["batch_size"]
        self.d_tau = conf["d_tau"]
        self.topK = conf["topk"]
        self.pretrain_file = conf["pretrain_file"]

        idx_value_dict = {}
        for idx, value in enumerate(train_matrix):
            if any(value.indices):
                idx_value_dict[idx] = value.indices

        self.user_pos_train = idx_value_dict

        self.num_users, self.num_items = dataset.num_users, dataset.num_items
        self.dataset = dataset
        self.sess = sess
        self.all_items = np.arange(self.num_items)

    def build_graph(self):
        with open(self.pretrain_file, "rb") as fin:
            pretrain_params = pickle.load(fin, encoding="latin")
        self.generator = GEN(self.num_items, self.num_users, self.factors_num, self.g_reg, param=pretrain_params,
                             learning_rate=self.lr)
        self.discriminator = DIS(self.num_items, self.num_users, self.factors_num, self.d_reg, param=None,
                                 learning_rate=self.lr)

    def get_train_data(self):
        users_list, items_list, labels_list = [], [], []
        train_users = list(self.user_pos_train.keys())
        with ThreadPoolExecutor() as executor:
            data = executor.map(self.get_train_data_one_user, train_users)
        data = list(data)
        for users, items, labels in data:
            users_list.extend(users)
            items_list.extend(items)
            labels_list.extend(labels)

        return users_list, items_list, labels_list

    def get_train_data_one_user(self, user):
        user_list, items_list, label_list = [], [], []
        pos = self.user_pos_train[user]

        rating = self.sess.run(self.generator.all_rating, {self.generator.u: [user]})
        rating = np.reshape(rating, [-1])
        rating = np.array(rating) / self.d_tau  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)
        neg = np.random.choice(self.all_items, size=len(pos), p=prob)
        for i, j in zip(pos, neg):
            user_list.append(user)
            items_list.append(i)
            label_list.append(1.0)

            user_list.append(user)
            items_list.append(j)
            label_list.append(0.0)
        return user_list, items_list, label_list

    def train_model(self):
        for _ in range(self.epochs):
            for _ in range(self.d_epoch):
                self.training_discriminator()
            for _ in range(self.g_epoch):
                self.training_generator()
                self.logger.info("%s" % (self.evaluate()))

    def training_discriminator(self):
        users_list, items_list, labels_list = self.get_train_data()
        data_iter = DataIterator(users_list, items_list, labels_list,
                                 batch_size=self.batch_size, shuffle=True)
        for bat_users, bat_items, bat_labels in data_iter:
            feed = {self.discriminator.u: bat_users,
                    self.discriminator.i: bat_items,
                    self.discriminator.label: bat_labels}
            self.sess.run(self.discriminator.d_updates, feed_dict=feed)

    def training_generator(self):
        for user, pos in self.user_pos_train.items():
            sample_lambda = 0.2
            rating = self.sess.run(self.generator.all_logits, {self.generator.u: user})
            exp_rating = np.exp(rating)
            prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

            pn = (1 - sample_lambda) * prob
            pn[pos] += sample_lambda * 1.0 / len(pos)
            # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta

            sample = np.random.choice(self.all_items, 2 * len(pos), p=pn)
            ###########################################################################
            # Get reward and adapt it with importance sampling
            ###########################################################################
            feed = {self.discriminator.u: user, self.discriminator.i: sample}
            reward = self.sess.run(self.discriminator.reward, feed_dict=feed)
            reward = reward * prob[sample] / pn[sample]
            ###########################################################################
            # Update G
            ###########################################################################
            feed = {self.generator.u: user, self.generator.i: sample, self.generator.reward: reward}
            self.sess.run(self.generator.gan_updates, feed_dict=feed)

    def evaluate(self):
        self._cur_user_embedding, self._cur_item_embedding, self._cur_item_bias = self.sess.run(self.generator.g_params)
        return self.evaluator.evaluate(self)

    def predict(self, users, items):
        user_embedding = self._cur_user_embedding[users]
        item_embedding = self._cur_item_embedding
        item_bias = self._cur_item_bias

        all_ratings = np.matmul(user_embedding, item_embedding.T) + item_bias
        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]
        return all_ratings
