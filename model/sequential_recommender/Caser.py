"""
Paper: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding
Author: Jiaxi Tang, and Ke Wang
Reference: https://github.com/graytowne/caser_pytorch
@author: Zhongchuan Sun
"""

import numpy as np
from model.AbstractRecommender import SeqAbstractRecommender
from util import DataIterator
from util.tool import csr_to_user_dict_bytime
import tensorflow as tf
from util import batch_randint_choice
from util import pad_sequences


class Caser(SeqAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(Caser, self).__init__(dataset, conf)
        self.dataset = dataset
        self.users_num, self.items_num = dataset.train_matrix.shape

        self.lr = conf["lr"]
        self.l2_reg = conf["l2_reg"]
        self.factors_num = conf["factors_num"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]
        self.seq_L = conf["seq_L"]
        self.seq_T = conf["seq_T"]
        self.nv = conf["nv"]
        self.nh = conf["nh"]
        self.dropout = conf["dropout"]
        self.neg_samples = conf["neg_samples"]

        self.sess = sess

    def _create_variable(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.item_seq_ph = tf.placeholder(tf.int32, [None, self.seq_L], name="item_seq")
        self.item_pos_ph = tf.placeholder(tf.int32, [None, self.seq_T], name="item_pos")
        self.item_neg_ph = tf.placeholder(tf.int32, [None, self.neg_samples], name="item_neg")
        self.is_training = tf.placeholder(tf.bool, name="training_flag")

        l2_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)

        self.user_embeddings = tf.get_variable('user_embeddings', dtype=tf.float32,
                                               shape=[self.users_num, self.factors_num],
                                               regularizer=l2_regularizer)

        item_embeddings = tf.get_variable('seq_item_embeddings', dtype=tf.float32,
                                          shape=[self.items_num, self.factors_num],
                                          regularizer=l2_regularizer)
        zero_pad = tf.zeros([1, self.factors_num], name="padding")
        self.seq_item_embeddings = tf.concat([item_embeddings, zero_pad], axis=0)

        self.conv_v = tf.layers.Conv2D(self.nv, [self.seq_L, 1])
        self.conv_h = [tf.layers.Conv2D(self.nh, [i, self.factors_num], activation="relu")
                       for i in np.arange(1, self.seq_L + 1)]

        self.fc1_ly = tf.layers.Dense(self.factors_num, activation="relu")
        self.dropout_ly = tf.layers.Dropout(self.dropout)

        # predication embedding
        self.item_embeddings = tf.get_variable('item_embeddings', dtype=tf.float32,
                                               shape=[self.items_num, self.factors_num * 2],
                                               regularizer=l2_regularizer)
        self.item_biases = tf.get_variable('item_biases', dtype=tf.float32, shape=[self.items_num],
                                           initializer=tf.initializers.zeros(), regularizer=l2_regularizer)

    def build_graph(self):
        self._create_variable()
        # embedding lookup
        batch_size = tf.shape(self.item_seq_ph)[0]
        item_embs = tf.nn.embedding_lookup(self.seq_item_embeddings, self.item_seq_ph)  # (b, L, d)
        item_embs = tf.expand_dims(item_embs, axis=3)  # (b, L, d, 1)

        # vertical conv layer
        out_v = self.conv_v(item_embs)  # (b, 1, d, nv)
        out_v = tf.reshape(out_v, shape=[batch_size, self.nv*self.factors_num])  # (b, nv*d)

        # horizontal conv layer
        out_hs = []
        for conv_h in self.conv_h:
            conv_out = conv_h(item_embs)  # (b, ?, 1, nh)
            conv_out = tf.squeeze(conv_out, axis=2)  # (b, ?, nh)
            pool_out = tf.reduce_max(conv_out, axis=1)  # (b, nh)
            out_hs.append(pool_out)
        out_h = tf.concat(out_hs, axis=1)  # (b, nh*L)

        out = tf.concat([out_v, out_h], axis=1)  # (b, nv*d+nh*L)
        # apply dropout
        out = self.dropout_ly(out, training=self.is_training)

        # fully-connected Layers
        z = self.fc1_ly(out)  # (b, d)

        # rating calculation
        user_embs = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)  # (b, d)
        user_embs = tf.concat([z, user_embs], axis=1)  # (b, 2d)
        user_embs = tf.expand_dims(user_embs, axis=1)  # (b, 1, 2d)

        tar_item = tf.concat([self.item_pos_ph, self.item_neg_ph], axis=-1)  # (b, 2T)
        tar_item_embs = tf.nn.embedding_lookup(self.item_embeddings, tar_item)  # (b, 2T, 2d)
        tar_item_bias = tf.gather(self.item_biases, tar_item)  # (b, 2T)
        logits = tf.squeeze(tf.matmul(user_embs, tar_item_embs, transpose_b=True), axis=1) + tar_item_bias  # (b, 2T)
        pos_logits, neg_logits = tf.split(logits, [self.seq_T, self.neg_samples], axis=1)

        # loss
        pos_loss = tf.reduce_mean(-tf.log(tf.sigmoid(pos_logits) + 1e-24))
        neg_loss = tf.reduce_mean(-tf.log(1 - tf.sigmoid(neg_logits) + 1e-24))
        loss = pos_loss + neg_loss
        try:
            reg_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss = loss + reg_losses
        except:
            pass

        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        # for predict/test
        user_embs = tf.squeeze(user_embs, axis=1)  # (b, 2d)
        self.all_logits = tf.matmul(user_embs, self.item_embeddings, transpose_b=True)  # (b, items_num)

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        self.user_pos_train = csr_to_user_dict_bytime(self.dataset.time_matrix, self.dataset.train_matrix)
        users_list, item_seq_list, item_pos_list = self._generate_sequences()
        for epoch in range(self.epochs):
            item_neg_list = self._sample_negative(users_list)
            data = DataIterator(users_list, item_seq_list, item_pos_list, item_neg_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_user, bat_item_seq, bat_item_pos, bat_item_neg in data:
                feed = {self.user_ph: bat_user,
                        self.item_seq_ph: bat_item_seq,
                        self.item_pos_ph: bat_item_pos,
                        self.item_neg_ph: bat_item_neg,
                        self.is_training: True}

                self.sess.run(self.train_opt, feed_dict=feed)

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _generate_sequences(self):
        self.user_test_seq = {}
        users_list, item_seq_list, item_pos_list = [], [], []
        seq_len = self.seq_L + self.seq_T
        uni_users = np.unique(list(self.user_pos_train.keys()))
        for user in uni_users:
            seq_items = self.user_pos_train[user]
            if len(seq_items) - seq_len >= 0:
                for i in range(len(seq_items), 0, -1):
                    if i-seq_len >= 0:
                        seq_i = seq_items[i-seq_len:i]
                        if user not in self.user_test_seq:
                            self.user_test_seq[user] = seq_i[-self.seq_L:]
                        users_list.append(user)
                        item_seq_list.append(seq_i[:self.seq_L])
                        item_pos_list.append(seq_i[-self.seq_T:])
                    else:
                        break
            else:
                seq_items = np.reshape(seq_items, newshape=[1, -1]).astype(np.int32)
                seq_items = pad_sequences(seq_items, value=self.items_num, max_len=seq_len,
                                          padding='pre', truncating='pre')
                seq_i = np.reshape(seq_items, newshape=[-1])
                if user not in self.user_test_seq:
                    self.user_test_seq[user] = seq_i[-self.seq_L:]
                users_list.append(user)
                item_seq_list.append(seq_i[:self.seq_L])
                item_pos_list.append(seq_i[-self.seq_T:])
        return users_list, item_seq_list, item_pos_list

    def _sample_negative(self, users_list):
        neg_items_list = []
        user_neg_items_dict = {}
        all_uni_user, all_counts = np.unique(users_list, return_counts=True)
        user_count = DataIterator(all_uni_user, all_counts, batch_size=1024, shuffle=False)
        for bat_users, bat_counts in user_count:
            n_neg_items = [c*self.neg_samples for c in bat_counts]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.items_num, n_neg_items, replace=True, exclusion=exclusion)
            for u, neg in zip(bat_users, bat_neg):
                user_neg_items_dict[u] = neg

        for u, c in zip(all_uni_user, all_counts):
            neg_items = np.reshape(user_neg_items_dict[u], newshape=[c, self.neg_samples])
            neg_items_list.extend(neg_items)
        return neg_items_list

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, items=None):
        users = DataIterator(users, batch_size=512, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            bat_seq = [self.user_test_seq[u] for u in bat_user]
            feed = {self.user_ph: bat_user,
                    self.item_seq_ph: bat_seq,
                    self.is_training: False}
            bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
            all_ratings.extend(bat_ratings)
        all_ratings = np.array(all_ratings, dtype=np.float32)

        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]

        return all_ratings
