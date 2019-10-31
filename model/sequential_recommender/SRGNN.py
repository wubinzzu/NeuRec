"""
Paper: Session-Based Recommendation with Graph Neural Networks
Author: Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan
Reference: https://github.com/CRIPAC-DIG/SR-GNN
@author: Zhongchuan Sun
"""

import numpy as np
from model.AbstractRecommender import SeqAbstractRecommender
from util import DataIterator
from util.Tool import csr_to_user_dict_bytime
import tensorflow as tf
from util import pad_sequences
from util.Logger import logger
import math


class SRGNN(SeqAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(SRGNN, self).__init__(dataset, conf)
        self.dataset = dataset
        self.train_matrix = self.dataset.train_matrix

        self.users_num, self.items_num = self.train_matrix.shape

        self.lr = conf["lr"]
        self.reg = conf["reg"]
        self.hidden_size = conf["hidden_size"]
        self.step = conf["step"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]

        self.all_items = np.arange(self.items_num)
        self.user_pos_train = csr_to_user_dict_bytime(self.train_matrix, self.dataset.time_matrix)

        self.sess = sess

    def _create_variable(self):
        self.items_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None], name="items")
        self.alias_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None], name="alias")
        self.mask_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None], name="mask")

        self.tar_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="target")
        self.adj_in_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None], name="adj_in")
        self.adj_out_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None], name="adj_out")

        stdv = 1.0 / math.sqrt(self.hidden_size)
        initializer = tf.initializers.random_uniform(-stdv, stdv)

        # the last row is used to padding
        item_embeddings = tf.get_variable("item_embeddings", shape=[self.items_num, self.hidden_size],
                                          dtype=tf.float32, initializer=initializer)
        zero_pad = tf.zeros([1, self.hidden_size], name="padding")
        self.item_embeddings = tf.concat([item_embeddings, zero_pad], axis=0)

        self.W_in = tf.get_variable('W_in', shape=[self.hidden_size, self.hidden_size],
                                    dtype=tf.float32, initializer=initializer)
        self.b_in = tf.get_variable('b_in', [self.hidden_size], dtype=tf.float32, initializer=initializer)
        self.W_out = tf.get_variable('W_out', [self.hidden_size, self.hidden_size],
                                     dtype=tf.float32, initializer=initializer)
        self.b_out = tf.get_variable('b_out', [self.hidden_size], dtype=tf.float32, initializer=initializer)

        self.nasr_w1 = tf.get_variable('nasr_w1', [self.hidden_size, self.hidden_size],
                                       dtype=tf.float32, initializer=initializer)
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.hidden_size, self.hidden_size],
                                       dtype=tf.float32, initializer=initializer)
        self.nasr_v = tf.get_variable('nasrv', [1, self.hidden_size], dtype=tf.float32, initializer=initializer)
        self.nasr_b = tf.get_variable('nasr_b', [self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer())

        self.B = tf.get_variable('B', [2 * self.hidden_size, self.hidden_size],
                                 dtype=tf.float32, initializer=initializer)

    def _node_gnn(self):  # node embedding
        fin_state = tf.nn.embedding_lookup(self.item_embeddings, self.items_ph)  # (b, 6, d)
        gru_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        for i in range(self.step):
            fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.hidden_size])  # (b, 6, d)
            fin_state = tf.reshape(fin_state, [-1, self.hidden_size])  # (b*6, d)

            # linearly transform and convolution for 'in' item
            state_in = tf.matmul(fin_state, self.W_in) + self.b_in  # (b*6, d)
            fin_state_in = tf.reshape(state_in, [self.batch_size, -1, self.hidden_size])  # (b, 6, d)
            fin_state_in = tf.matmul(self.adj_in_ph, fin_state_in)  # (b, 6, d)

            # linearly transform and convolution for 'out' item
            state_out = tf.matmul(fin_state, self.W_out) + self.b_out  # (b*6, d)
            fin_state_out = tf.reshape(state_out, [self.batch_size, -1, self.hidden_size])  # (b, 6, d)
            fin_state_out = tf.matmul(self.adj_out_ph, fin_state_out)  # (b, 6, d)

            # concat
            av = tf.concat([fin_state_in, fin_state_out], axis=-1)  # (b, 6, 2d)

            av = tf.reshape(av, [-1, 2 * self.hidden_size])  # (b*6, 2d)
            av_input = tf.expand_dims(av, axis=1)  # (b*6, 1, 2d)
            state_input = tf.reshape(fin_state, [-1, self.hidden_size])  # (b*6, d)
            state_output, fin_state = tf.nn.dynamic_rnn(gru_cell, av_input, initial_state=state_input)

        return tf.reshape(fin_state, [self.batch_size, -1, self.hidden_size])  # (b, 6, d)

    def _session_embedding(self, re_embedding):  # generating session embeddings
        rm = tf.reduce_sum(self.mask_ph, 1)  # (b,), length of each session
        last_idx = tf.stack([tf.range(self.batch_size), tf.to_int32(rm) - 1], axis=1)  # (b, 2), index of last item in each session
        last_id = tf.gather_nd(self.alias_ph, last_idx)  # (b,), inner id of last item in each session

        last_idx = tf.stack([tf.range(self.batch_size), last_id], axis=1)  # (b, 2)
        last_h = tf.gather_nd(re_embedding, last_idx)  # (b, d), the embedding of last item of each session

        last = tf.matmul(last_h, self.nasr_w1)  # (b, d)
        last = tf.reshape(last, [self.batch_size, 1, -1])  # (b, 1, d)

        seq_h = [tf.nn.embedding_lookup(re_embedding[i], self.alias_ph[i]) for i in range(self.batch_size)]
        seq_h = tf.stack(seq_h, axis=0)  # (b, 16, d)
        seq_h = tf.reshape(seq_h, [-1, self.hidden_size])  # (b*16, d)
        seq = tf.matmul(seq_h, self.nasr_w2)  # (b*16, d)
        seq = tf.reshape(seq, [self.batch_size, -1, self.hidden_size])  # (b, 16, d)

        m = tf.nn.sigmoid(last + seq + self.nasr_b)  # (b, 16, d)
        m = tf.reshape(m, [-1, self.hidden_size])  # (b*16, d)

        coef = tf.matmul(m, self.nasr_v, transpose_b=True)  # (b*16, 1)
        mask = tf.cast(tf.reshape(self.mask_ph, [-1, 1]), dtype=tf.float32)  # (b*16, 1)
        coef = tf.multiply(coef, mask)  # (b*16, 1)
        coef = tf.reshape(coef, [self.batch_size, -1, 1])  # (b, 16, 1)
        seq_h = tf.reshape(seq_h, [self.batch_size, -1, self.hidden_size])  # (b, 16, d)
        s_g = tf.reduce_sum(coef * seq_h, 1)  # (b, d)
        s_1 = tf.reshape(last, [-1, self.hidden_size])  # (b, d)

        ma = tf.concat([s_1, s_g], -1)  # (b, 2d)
        s_h = tf.matmul(ma, self.B)  # (b, d)  the representation of a session
        return s_h

    def build_graph(self):
        self._create_variable()
        self.re_embeddings = self._node_gnn()
        sess_h = self._session_embedding(self.re_embeddings)

        item_embed = self.item_embeddings[:-1]  # (items_num, d), all item embedding without padding row
        self.logits = tf.matmul(sess_h, item_embed, transpose_b=True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar_ph, logits=self.logits))

        params_vars = tf.trainable_variables()
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in params_vars
                             if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']])

        loss = loss + self.reg*reg_loss
        self.update_opt = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def train_model(self):
        logger.info(self.evaluator.metrics_info())
        users_list, items_list = [], []
        for user, items in self.user_pos_train.items():
            users_list.append(user)
            items_list.append(items)
        for epoch in range(self.epochs):
            data = DataIterator(users_list, items_list, batch_size=self.batch_size, shuffle=True, drop_last=True)
            for bat_users, bat_items in data:
                targets = [items[-1] for items in bat_items]
                sess_list = [items[:-1] for items in bat_items]
                bat_adj_in, bat_adj_out, bat_alias, bat_items, bat_mask = self._build_session_graph(sess_list)
                feed = {self.items_ph: bat_items,
                        self.adj_in_ph: bat_adj_in,
                        self.adj_out_ph: bat_adj_out,
                        self.alias_ph: bat_alias,
                        self.mask_ph: bat_mask,
                        self.tar_ph: targets}

                self.sess.run(self.update_opt, feed_dict=feed)

            logger.info("epoch %d:\t%s" % (epoch, self.evaluate_model()))

    def _build_session_graph(self, bat_items):
        A_in, A_out, alias_inputs, items = [], [], [], []
        all_mask = [[1]*len(items) for items in bat_items]
        bat_items = pad_sequences(bat_items, value=self.items_num)

        unique_nodes = [np.unique(items).tolist() for items in bat_items]
        max_n_node = np.max([len(nodes) for nodes in unique_nodes])
        for u_seq, u_node, mask in zip(bat_items, unique_nodes, all_mask):
            adj_mat = np.zeros((max_n_node, max_n_node))
            id_map = {node: idx for idx, node in enumerate(u_node)}
            if len(u_seq) > 1:
                alias_previous = [id_map[i] for i in u_seq[:len(mask)-1]]
                alias_next = [id_map[i] for i in u_seq[1:len(mask)]]
                adj_mat[alias_previous, alias_next] = 1

            u_sum_in = np.sum(adj_mat, axis=0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(adj_mat, u_sum_in)

            u_sum_out = np.sum(adj_mat, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(adj_mat.transpose(), u_sum_out)

            A_in.append(u_A_in)
            A_out.append(u_A_out)
            alias_inputs.append([id_map[i] for i in u_seq])

        all_mask = pad_sequences(all_mask, value=0)
        return A_in, A_out, alias_inputs, bat_items, all_mask

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, items):
        users = DataIterator(users, batch_size=self.batch_size, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            cur_batch_size = len(bat_user)
            bat_items = [self.user_pos_train[user] for user in bat_user]
            bat_adj_in, bat_adj_out, bat_alias, bat_items, bat_mask = self._build_session_graph(bat_items)
            if cur_batch_size < self.batch_size:  # padding
                bat_adj_in += [bat_adj_in[-1]]*(self.batch_size-cur_batch_size)
                bat_adj_out += [bat_adj_out[-1]] * (self.batch_size - cur_batch_size)
                bat_alias += [bat_alias[-1]] * (self.batch_size - cur_batch_size)
                bat_items += [bat_items[-1]] * (self.batch_size - cur_batch_size)
                bat_mask += [bat_mask[-1]] * (self.batch_size - cur_batch_size)

            feed = {self.items_ph: bat_items,
                    self.adj_in_ph: bat_adj_in,
                    self.adj_out_ph: bat_adj_out,
                    self.alias_ph: bat_alias,
                    self.mask_ph: bat_mask}
            bat_ratings = self.sess.run(self.logits, feed_dict=feed)
            all_ratings.extend(bat_ratings[:cur_batch_size])
        all_ratings = np.array(all_ratings)
        if items is not None:
            all_ratings = [all_ratings[idx][u_item] for idx, u_item in enumerate(items)]

        return all_ratings
