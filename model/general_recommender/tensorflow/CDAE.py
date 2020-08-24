"""
Paper: Collaborative Denoising Auto-Encoder for Top-N Recommender Systems
Author: Yao Wu, Christopher DuBois, Alice X. Zheng, and Martin Ester
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["CDAE"]

import tensorflow as tf
from model.base import AbstractRecommender
from util.tensorflow import inner_product, l2_loss
from util.tensorflow import pointwise_loss
from util.tensorflow import get_initializer, get_session
from util.common import Reduction
import numpy as np
from util.tensorflow import dropout_sparse
from reckit import DataIterator
from reckit import randint_choice


class CDAE(AbstractRecommender):
    def __init__(self, config):
        super(CDAE, self).__init__(config)
        self.emb_size = config["hidden_dim"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.dropout = config["dropout"]
        self.loss_func = config["loss_func"]

        self.num_neg = config["num_neg"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.param_init = config["param_init"]

        self.train_csr_mat = self.dataset.train_data.to_csr_matrix()
        self.train_csr_mat.data[:] = 1.0
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        if config["hidden_act"] == "identity":
            self.hidden_act = tf.identity
        elif config["hidden_act"] == "sigmoid":
            self.hidden_act = tf.sigmoid
        else:
            raise ValueError(f"hidden activate function {config['hidden_act']} is invalid.")

        self._build_model()
        self.sess = get_session(config["gpu_mem"])

    def _create_variable(self):
        self.users_ph = tf.placeholder(tf.int32, [None], name="user")
        self.sp_mat_ph = tf.sparse_placeholder(tf.float32, [None, self.num_items], name="sp_mat")
        self.items_ph = tf.placeholder(tf.int32, [None], name="items")
        self.labels_ph = tf.placeholder(tf.float32, [None], name="labels")
        self.remap_idx_ph = tf.placeholder(tf.int32, [None], name="remap_idx")
        self.dropout_ph = tf.placeholder(tf.float32, name="dropout")
        self.noise_shape_ph = tf.placeholder(tf.int32, name="noise_shape")

        # embedding layers
        init = get_initializer(self.param_init)
        zero_init = get_initializer("zeros")
        self.user_embeddings = tf.Variable(init([self.num_users, self.emb_size]), name="user_embedding")
        self.en_embeddings = tf.Variable(init([self.num_items, self.emb_size]), name="en_embeddings")
        self.en_offset = tf.Variable(zero_init([self.emb_size]), name="en_offset")
        self.de_embeddings = tf.Variable(init([self.num_items, self.emb_size]), name="de_embeddings")
        self.de_bias = tf.Variable(zero_init([self.num_items]), name="item_bias")

    def _encoding(self, user_ids, sp_item_mat):
        corruption = dropout_sparse(sp_item_mat, 1-self.dropout, self.noise_shape_ph)
        hidden = tf.sparse_tensor_dense_matmul(corruption, self.en_embeddings)

        user_embs = tf.nn.embedding_lookup(self.user_embeddings, user_ids)
        en_offset = tf.reshape(self.en_offset, shape=[1, -1])
        hidden = hidden + user_embs + en_offset
        hidden = self.hidden_act(hidden)
        return hidden

    def _build_model(self):
        self._create_variable()
        hidden_ori = self._encoding(self.users_ph, self.sp_mat_ph)  # (b,d)

        # decoding
        de_item_embs = tf.nn.embedding_lookup(self.de_embeddings, self.items_ph)  # (l,d)
        de_bias = tf.gather(self.de_bias, self.items_ph)  # (l,d)
        hidden = tf.nn.embedding_lookup(hidden_ori, self.remap_idx_ph)
        ratings = inner_product(hidden, de_item_embs) + de_bias

        # reg loss
        item_ids, _ = tf.unique(self.items_ph)
        reg_loss = l2_loss(tf.nn.embedding_lookup(self.en_embeddings, item_ids), self.en_offset,
                           tf.nn.embedding_lookup(self.user_embeddings, self.users_ph),
                           tf.nn.embedding_lookup(self.de_embeddings, item_ids),
                           tf.gather(self.de_bias, item_ids))

        model_loss = pointwise_loss(self.loss_func, ratings, self.labels_ph, reduction=Reduction.SUM)

        final_loss = model_loss + self.reg * reg_loss

        self.train_opt = tf.train.AdamOptimizer(self.lr).minimize(final_loss, name="train_opt")

        # for evaluation
        self.batch_ratings = tf.matmul(hidden_ori, self.de_embeddings, transpose_b=True) + self.de_bias

    def train_model(self):
        train_users = [user for user in range(self.num_users) if self.train_csr_mat[user].nnz]
        user_iter = DataIterator(train_users, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            for bat_users in user_iter:
                bat_sp_mat = self.train_csr_mat[bat_users]
                bat_items = []
                bat_labels = []
                bat_idx = []  # used to decoder
                for idx, _ in enumerate(bat_users):
                    pos_items = bat_sp_mat[idx].indices
                    neg_items = randint_choice(self.num_items, size=bat_sp_mat[idx].nnz*self.num_neg,
                                               replace=True, exclusion=pos_items)
                    neg_items = np.unique(neg_items)
                    bat_sp_mat[idx, neg_items] = 1

                    bat_items.append(pos_items)
                    bat_labels.append(np.ones_like(pos_items, dtype=np.float32))
                    bat_items.append(neg_items)
                    bat_labels.append(np.zeros_like(neg_items, dtype=np.float32))
                    bat_idx.append(np.full(len(pos_items)+len(neg_items), idx, dtype=np.int32))

                bat_items = np.concatenate(bat_items)
                bat_labels = np.concatenate(bat_labels)
                bat_idx = np.concatenate(bat_idx)
                bat_users = np.asarray(bat_users)

                coo = bat_sp_mat.tocoo().astype(np.float32)
                indices = np.asarray([coo.row, coo.col]).transpose()

                feed = {self.users_ph: bat_users,
                        self.remap_idx_ph: bat_idx,
                        self.items_ph: bat_items,
                        self.sp_mat_ph: (indices, coo.data, coo.shape),
                        self.labels_ph: bat_labels,
                        self.dropout_ph: self.dropout,
                        self.noise_shape_ph: bat_sp_mat.nnz}
                self.sess.run(self.train_opt, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users = np.asarray(users)
        sp_mat = self.train_csr_mat[users]

        coo = sp_mat.tocoo().astype(np.float32)
        indices = np.asarray([coo.row, coo.col]).transpose()

        feed = {self.users_ph: users,
                self.sp_mat_ph: (indices, coo.data, coo.shape),
                self.dropout_ph: 0.0,
                self.noise_shape_ph: sp_mat.nnz}
        all_ratings = self.sess.run(self.batch_ratings, feed_dict=feed)
        return all_ratings
