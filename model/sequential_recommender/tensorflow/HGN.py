"""
Paper: Hierarchical Gating Networks for Sequential Recommendation
Author: Chen Ma, Peng Kang, and Xue Liu
Reference: https://github.com/allenjack/HGN
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["HGN"]

import tensorflow as tf
from model.base import AbstractRecommender
from util.tensorflow import bpr_loss
from util.tensorflow import get_initializer, get_session
from util.common import Reduction
from data import TimeOrderPairwiseSampler


class HGN(AbstractRecommender):
    def __init__(self, config):
        super(HGN, self).__init__(config)
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.seq_L = config["L"]
        self.seq_T = config["T"]
        self.emb_size = config["embedding_size"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

        self.param_init = config["param_init"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_dict = self.dataset.train_data.to_user_dict(by_time=True)
        self.pad_idx = self.num_items

        self.user_truncated_seq = self.dataset.train_data.to_truncated_seq_dict(self.seq_L,
                                                                                pad_value=self.pad_idx,
                                                                                padding='pre', truncating='pre')

        self._build_model()
        self.sess = get_session(config["gpu_mem"])

    def _create_variable(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.item_seqs_ph = tf.placeholder(tf.int32, [None, self.seq_L], name="item_seqs")
        self.pos_item_ph = tf.placeholder(tf.int32, [None, self.seq_T], name="pos_item")
        self.neg_item_ph = tf.placeholder(tf.int32, [None, self.seq_T], name="neg_item")
        self.predict_item = tf.concat([self.pos_item_ph, self.neg_item_ph], axis=1, name="item_to_predict")

        l2_reg = tf.contrib.layers.l2_regularizer(self.reg)
        init = get_initializer(self.param_init)
        he_init = get_initializer("he_uniform")
        xavier_init = get_initializer("xavier_uniform")
        zero_init = get_initializer("zeros")

        self.user_embeddings = tf.get_variable("user_embeddings", dtype=tf.float32, initializer=init,
                                               shape=[self.num_users, self.emb_size], regularizer=l2_reg)
        item_embeddings = tf.get_variable("item_embeddings", dtype=tf.float32, initializer=init,
                                          shape=[self.num_items, self.emb_size], regularizer=l2_reg)
        zero_pad = tf.Variable(zero_init([1, self.emb_size]), trainable=False, dtype=tf.float32, name="pad1")
        self.item_embeddings = tf.concat([item_embeddings, zero_pad], axis=0)

        self.feature_gate_item = tf.layers.Dense(self.emb_size, name="feature_gate_item",
                                                 kernel_initializer=he_init, bias_initializer=zero_init,
                                                 kernel_regularizer=l2_reg, bias_regularizer=l2_reg)
        self.feature_gate_user = tf.layers.Dense(self.emb_size, name="feature_gate_user",
                                                 kernel_initializer=he_init, bias_initializer=zero_init,
                                                 kernel_regularizer=l2_reg, bias_regularizer=l2_reg)

        self.instance_gate_item = tf.get_variable("instance_gate_item", dtype=tf.float32, initializer=xavier_init,
                                                  shape=[self.emb_size, 1], regularizer=l2_reg)
        self.instance_gate_user = tf.get_variable("instance_gate_user", dtype=tf.float32, initializer=xavier_init,
                                                  shape=[self.emb_size, self.seq_L], regularizer=l2_reg)

        W2 = tf.get_variable("W2", dtype=tf.float32, initializer=init,
                             shape=[self.num_items, self.emb_size], regularizer=l2_reg)
        self.W2 = tf.concat([W2, zero_pad], axis=0)
        b2 = tf.get_variable("b2", dtype=tf.float32, initializer=zero_init,
                             shape=[self.num_items], regularizer=l2_reg)
        zero_pad = tf.Variable(zero_init([1]), trainable=False, dtype=tf.float32, name="pad2")
        self.b2 = tf.concat([b2, zero_pad], axis=0)

    def _forward_user(self, user_emb, item_embs):  # (b,l,d), (b,d)
        gate = tf.sigmoid(self.feature_gate_item(item_embs) +
                          tf.expand_dims(self.feature_gate_user(user_emb), axis=1))  # (b,l,d)

        # feature gating
        gated_item = tf.multiply(item_embs, gate)  # (b,l,d)

        # instance gating
        term1 = tf.matmul(gated_item, tf.expand_dims(self.instance_gate_item, axis=0))  # (b,l,d)x(1,d,1)->(b,l,1)
        term2 = tf.matmul(user_emb, self.instance_gate_user)  # (b,d)x(d,l)->(b,l)

        instance_score = tf.sigmoid(tf.squeeze(term1) + term2)  # (b,l)
        union_out = tf.multiply(gated_item, tf.expand_dims(instance_score, axis=2))  # (b,l,d)
        union_out = tf.reduce_sum(union_out, axis=1)  # (b,d)
        instance_score = tf.reduce_sum(instance_score, axis=1, keep_dims=True)
        union_out = union_out / instance_score  # (b,d)
        return union_out  # (b,d)

    def _train_rating(self, user_emb, item_embs, union_out):
        w2 = tf.nn.embedding_lookup(self.W2, self.predict_item)  # (b,2t,d)
        b2 = tf.gather(self.b2, self.predict_item)  # (b,2t)

        # MF
        term3 = tf.squeeze(tf.matmul(w2, tf.expand_dims(user_emb, axis=2)))  # (b,2t,d)x(b,d,1)->(b,2t,1)->(b,2t)
        res = b2 + term3  # (b,2t)

        # union-level
        term4 = tf.matmul(tf.expand_dims(union_out, axis=1), w2, transpose_b=True)  # (b,1,d)x(b,d,2l)->(b,1,2l)
        res += tf.squeeze(term4)  # (b,2t)

        # item-item product
        rel_score = tf.matmul(item_embs, w2, transpose_b=True)  # (b,l,d)x(b,d,2t)->(b,l,2t)
        rel_score = tf.reduce_sum(rel_score, axis=1)  # (b,2t)

        res += rel_score  # (b,2t)
        return res

    def _test_rating(self, user_emb, item_embs, union_out):
        # for testing
        w2 = self.W2  # (n,d)
        b2 = self.b2  # (n,)

        # MF
        res = tf.matmul(user_emb, w2, transpose_b=True) + b2  # (b,d)x(d,n)->(b,n)

        # union-level
        res += tf.matmul(union_out, w2, transpose_b=True)  # (b,d)x(d,n)->(b,n)

        # item-item product
        rel_score = tf.matmul(item_embs, w2, transpose_b=True)  # (b,l,d)x(d,n)->(b,l,n)
        rel_score = tf.reduce_sum(rel_score, axis=1)  # (b,n)

        res += rel_score
        return res  # (b,n)

    def _build_model(self):
        self._create_variable()

        item_embs = tf.nn.embedding_lookup(self.item_embeddings, self.item_seqs_ph)  # (b,l,d)
        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)  # (b,d)

        union_out = self._forward_user(user_emb, item_embs)  # (b,d)
        train_ratings = self._train_rating(user_emb, item_embs, union_out)  # (b,2t)

        pos_ratings, neg_ratings = tf.split(train_ratings, [self.seq_T, self.seq_T], axis=1)
        loss = bpr_loss(pos_ratings-neg_ratings, reduction=Reduction.SUM)

        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        final_loss = loss + self.reg*reg_loss

        self.train_opt = tf.train.AdamOptimizer(self.lr).minimize(final_loss)
        # to avoid NAN
        weights = self.feature_gate_item.trainable_weights + self.feature_gate_user.trainable_weights
        weights.extend([self.instance_gate_item, self.instance_gate_user])
        with tf.control_dependencies([self.train_opt]):
            self.train_opt = [tf.assign(weight, tf.clip_by_norm(weight, 1.0))
                              for weight in weights]

        # for testing
        self.bat_ratings = self._test_rating(user_emb, item_embs, union_out)  # (b,n)

    def train_model(self):
        data_iter = TimeOrderPairwiseSampler(self.dataset.train_data, len_seqs=self.seq_L,
                                             len_next=self.seq_T, pad=self.pad_idx,
                                             num_neg=self.seq_T,
                                             batch_size=self.batch_size,
                                             shuffle=True, drop_last=False)

        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            for bat_user, bat_item_seq, bat_item_pos, bat_item_neg in data_iter:
                feed = {self.user_ph: bat_user,
                        self.item_seqs_ph: bat_item_seq,
                        self.pos_item_ph: bat_item_pos,
                        self.neg_item_ph: bat_item_neg,
                        }
                self.sess.run(self.train_opt, feed_dict=feed)

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, neg_items=None):
        bat_seq = [self.user_truncated_seq[u] for u in users]
        feed = {self.user_ph: users,
                self.item_seqs_ph: bat_seq
                }
        bat_ratings = self.sess.run(self.bat_ratings, feed_dict=feed)
        return bat_ratings
