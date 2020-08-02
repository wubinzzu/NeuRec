__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Caser"]

import tensorflow as tf
from model.base import AbstractRecommender
from util.tensorflow import sigmoid_cross_entropy
from util.tensorflow import get_initializer
from util.common import Reduction
from data import TimeOrderPairwiseSampler


class Caser(AbstractRecommender):
    def __init__(self, config):
        super(Caser, self).__init__(config)

        self.lr = config["lr"]
        self.l2_reg = config["l2_reg"]
        self.emb_size = config["emb_size"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.seq_L = config["seq_L"]
        self.seq_T = config["seq_T"]
        self.nv = config["nv"]
        self.nh = config["nh"]
        self.dropout = config["dropout"]
        self.neg_samples = config["neg_samples"]

        self.param_init = config["param_init"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.pad_idx = self.num_items
        self.num_items += 1

        self.user_truncated_seq = self.dataset.train_data.to_truncated_seq_dict(self.seq_L,
                                                                                pad_value=self.pad_idx,
                                                                                padding='pre', truncating='pre')

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = config["gpu_mem"]
        self.sess = tf.Session(config=tf_config)
        self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _create_variable(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.item_seq_ph = tf.placeholder(tf.int32, [None, self.seq_L], name="item_seq")
        self.item_pos_ph = tf.placeholder(tf.int32, [None, self.seq_T], name="item_pos")
        self.item_neg_ph = tf.placeholder(tf.int32, [None, self.neg_samples], name="item_neg")
        self.is_training = tf.placeholder(tf.bool, name="training_flag")

        l2_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
        initializer = get_initializer(self.param_init)
        self.user_embeddings = tf.get_variable('user_embeddings', dtype=tf.float32,
                                               initializer=initializer,
                                               shape=[self.num_users, self.emb_size],
                                               regularizer=l2_regularizer)

        self.seq_item_embeddings = tf.get_variable('seq_item_embeddings', dtype=tf.float32,
                                                   initializer=initializer,
                                                   shape=[self.num_items, self.emb_size],
                                                   regularizer=l2_regularizer)

        self.conv_v = tf.layers.Conv2D(self.nv, [self.seq_L, 1])
        self.conv_h = [tf.layers.Conv2D(self.nh, [i, self.emb_size], activation="relu")
                       for i in range(1, self.seq_L + 1)]

        self.fc1_ly = tf.layers.Dense(self.emb_size, activation="relu")
        self.dropout_ly = tf.layers.Dropout(self.dropout)

        # predication embedding
        self.item_embeddings = tf.get_variable('item_embeddings', dtype=tf.float32,
                                               initializer=initializer,
                                               shape=[self.num_items, self.emb_size * 2],
                                               regularizer=l2_regularizer)
        self.item_biases = tf.get_variable('item_biases', dtype=tf.float32, shape=[self.num_items],
                                           initializer=tf.initializers.zeros(),
                                           regularizer=l2_regularizer)

    def _build_model(self):
        self._create_variable()
        # embedding lookup
        batch_size = tf.shape(self.item_seq_ph)[0]
        item_embs = tf.nn.embedding_lookup(self.seq_item_embeddings, self.item_seq_ph)  # (b, L, d)
        item_embs = tf.expand_dims(item_embs, axis=3)  # (b, L, d, 1)

        # vertical conv layer
        out_v = self.conv_v(item_embs)  # (b, 1, d, nv)
        out_v = tf.reshape(out_v, shape=[batch_size, self.nv * self.emb_size])  # (b, nv*d)

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
        ones = tf.ones_like(pos_logits, dtype=tf.float32)
        zeors = tf.zeros_like(neg_logits, dtype=tf.float32)
        pos_loss = sigmoid_cross_entropy(y_pre=pos_logits, y_true=ones, reduction=Reduction.MEAN)
        neg_loss = sigmoid_cross_entropy(y_pre=neg_logits, y_true=zeors, reduction=Reduction.MEAN)
        loss = pos_loss + neg_loss

        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = loss + reg_loss

        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        # for evaluation
        user_embs = tf.squeeze(user_embs, axis=1)  # (b, 2d)
        self.batch_ratings = tf.matmul(user_embs, self.item_embeddings, transpose_b=True)  # (b, items_num)

    def train_model(self):
        data_iter = TimeOrderPairwiseSampler(self.dataset.train_data, len_seqs=self.seq_L,
                                             len_next=self.seq_T, pad=self.pad_idx,
                                             num_neg=self.neg_samples,
                                             batch_size=self.batch_size,
                                             shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            for bat_users, bat_item_seqs, bat_pos_items, bat_neg_items in data_iter:
                feed = {self.user_ph: bat_users,
                        self.item_seq_ph: bat_item_seqs,
                        self.item_pos_ph: bat_pos_items,
                        self.item_neg_ph: bat_neg_items,
                        self.is_training: True}

                self.sess.run(self.train_opt, feed_dict=feed)

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, neg_items=None):
        bat_seq = [self.user_truncated_seq[u] for u in users]
        feed = {self.user_ph: users,
                self.item_seq_ph: bat_seq,
                self.is_training: False}
        all_ratings = self.sess.run(self.batch_ratings, feed_dict=feed)
        return all_ratings
