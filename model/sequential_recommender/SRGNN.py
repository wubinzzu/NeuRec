"""
Paper: Session-Based Recommendation with Graph Neural Networks
Author: Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan
Reference: https://github.com/CRIPAC-DIG/SR-GNN
@author: Zhongchuan Sun
"""

import numpy as np
from model.AbstractRecommender import SeqAbstractRecommender
from util import DataIterator
import tensorflow as tf
from util import pad_sequences
import math


class SRGNN(SeqAbstractRecommender):
    def __init__(self, sess, dataset, config):
        super(SRGNN, self).__init__(dataset, config)
        self.lr = config["lr"]
        self.L2 = config["L2"]
        self.hidden_size = config["hidden_size"]
        self.step = config["step"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.lr_dc = config["lr_dc"]
        self.lr_dc_step = config["lr_dc_step"]
        self.nonhybrid = config["nonhybrid"]
        self.max_seq_len = config["max_seq_len"]

        self.num_users, self.num_item = dataset.num_users, dataset.num_items
        self.user_pos_train = dataset.get_user_train_dict(by_time=True)

        self.train_seq = []
        self.train_tar = []
        for user, seqs in self.user_pos_train.items():
            for i in range(1, len(seqs)):
                self.train_seq.append(seqs[-i - self.max_seq_len:-i])
                self.train_tar.append(seqs[-i])

        self.sess = sess

    def _create_variable(self):
        self.mask_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None])
        self.alias_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])  # 给给每个输入重新
        self.item_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])  # 原始ID+padID, pad 在后面
        self.target_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

        self.adj_in_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])

        stdv = 1.0 / math.sqrt(self.hidden_size)
        w_init = tf.random_uniform_initializer(-stdv, stdv)
        self.nasr_w1 = tf.get_variable('nasr_w1', [self.hidden_size, self.hidden_size], dtype=tf.float32,
                                       initializer=w_init)
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.hidden_size, self.hidden_size], dtype=tf.float32,
                                       initializer=w_init)
        self.nasr_v = tf.get_variable('nasrv', [1, self.hidden_size], dtype=tf.float32, initializer=w_init)
        self.nasr_b = tf.get_variable('nasr_b', [self.hidden_size], dtype=tf.float32,
                                      initializer=tf.zeros_initializer())

        embedding = tf.get_variable(shape=[self.num_item, self.hidden_size], name='embedding',
                                    dtype=tf.float32, initializer=w_init)
        zero_pad = tf.zeros([1, self.hidden_size], name="padding")
        self.embedding = tf.concat([embedding, zero_pad], axis=0)

        self.W_in = tf.get_variable('W_in', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32,
                                    initializer=w_init)
        self.b_in = tf.get_variable('b_in', [self.hidden_size], dtype=tf.float32, initializer=w_init)
        self.W_out = tf.get_variable('W_out', [self.hidden_size, self.hidden_size], dtype=tf.float32,
                                     initializer=w_init)
        self.b_out = tf.get_variable('b_out', [self.hidden_size], dtype=tf.float32, initializer=w_init)

        self.B = tf.get_variable('B', [2 * self.hidden_size, self.hidden_size], initializer=w_init)

    def ggnn(self):
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item_ph)  # (b,l,d)
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.hidden_size])  # (b,l,d)
                fin_state_tmp = tf.reshape(fin_state, [-1, self.hidden_size])  # (b*l,d)

                fin_state_in = tf.reshape(tf.matmul(fin_state_tmp, self.W_in) + self.b_in,
                                          [self.batch_size, -1, self.hidden_size])  # (b,l,d)

                # fin_state_tmp = tf.reshape(fin_state, [-1, self.hidden_size])  # (b*l,d)
                fin_state_out = tf.reshape(tf.matmul(fin_state_tmp, self.W_out) + self.b_out,
                                           [self.batch_size, -1, self.hidden_size])  # (b,l,d)

                av_in = tf.matmul(self.adj_in_ph, fin_state_in)  # (b,l,d)
                av_out = tf.matmul(self.adj_out_ph, fin_state_out)  # (b,l,d)
                av = tf.concat([av_in, av_out], axis=-1)  # (b,l,2d)

                av = tf.expand_dims(tf.reshape(av, [-1, 2 * self.hidden_size]), axis=1)  # (b*l,1,2d)
                # fin_state_tmp = tf.reshape(fin_state, [-1, self.hidden_size])  # (b*l,d)

                state_output, fin_state = tf.nn.dynamic_rnn(cell, av, initial_state=fin_state_tmp)
        return tf.reshape(fin_state, [self.batch_size, -1, self.hidden_size])  # (b,l,d)

    def _session_embedding(self, re_embedding):
        # re_embedding  (b,l,d)
        rm = tf.reduce_sum(self.mask_ph, 1)  # (b,), length of each session
        last_idx = tf.stack([tf.range(self.batch_size), tf.to_int32(rm) - 1], axis=1)  # (b, 2) index of last item
        last_id = tf.gather_nd(self.alias_ph, last_idx)  # (b,) alias id of last item
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))  # (b,d) embedding of last item

        seq_h = [tf.nn.embedding_lookup(re_embedding[i], self.alias_ph[i]) for i in range(self.batch_size)]
        seq_h = tf.stack(seq_h, axis=0)  # batch_size*T*d
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.hidden_size]), self.nasr_w2)
        last = tf.reshape(last, [self.batch_size, 1, -1])
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.hidden_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.hidden_size]), self.nasr_v, transpose_b=True) * tf.reshape(self.mask_ph, [-1, 1])
        if not self.nonhybrid:
            ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1),
                            tf.reshape(last, [-1, self.hidden_size])], -1)
            sess_embedding = tf.matmul(ma, self.B)
        else:
            sess_embedding = tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1)

        return sess_embedding

    def build_graph(self):
        self._create_variable()
        with tf.variable_scope('ggnn_model', reuse=None):
            node_embedding = self.ggnn()
            sess_embedding = self._session_embedding(node_embedding)

        item_embedding = self.embedding[:-1]
        self.all_logits = tf.matmul(sess_embedding, item_embedding, transpose_b=True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_ph, logits=self.all_logits)
        loss = tf.reduce_mean(loss)

        vars = tf.trainable_variables()
        lossL2 = [tf.nn.l2_loss(v) for v in vars if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]
        loss_train = loss + self.L2 * tf.add_n(lossL2)

        global_step = tf.Variable(0)
        decay = self.lr_dc_step * len(self.train_seq) / self.batch_size
        learning_rate = tf.train.exponential_decay(self.lr, global_step=global_step, decay_steps=decay,
                                                   decay_rate=self.lr_dc, staircase=True)
        self.train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss_train, global_step=global_step)

    def train_model(self):
        train_seq_len = [(idx, len(seq)) for idx, seq in enumerate(self.train_seq)]
        train_seq_len = sorted(train_seq_len, key=lambda x: x[1], reverse=True)
        train_seq_index, _ = list(zip(*train_seq_len))

        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            for bat_index in self._shuffle_index(train_seq_index):
                item_seqs = [self.train_seq[idx] for idx in bat_index]
                bat_tars = [self.train_tar[idx] for idx in bat_index]
                bat_adj_in, bat_adj_out, bat_alias, bat_items, bat_mask = self._build_session_graph(item_seqs)
                feed = {self.target_ph: bat_tars,
                        self.item_ph: bat_items,
                        self.adj_in_ph: bat_adj_in,
                        self.adj_out_ph: bat_adj_out,
                        self.alias_ph: bat_alias,
                        self.mask_ph: bat_mask}

                self.sess.run(self.train_opt, feed_dict=feed)

            self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate_model()))

    def _shuffle_index(self, seq_index):
        index_chunks = DataIterator(seq_index, batch_size=self.batch_size*32, shuffle=False, drop_last=False)  # chunking
        index_chunks = list(index_chunks)
        index_chunks_iter = DataIterator(index_chunks, batch_size=1, shuffle=True, drop_last=False)  # shuffle index chunk
        for indexes in index_chunks_iter:
            indexes = indexes[0]
            indexes_iter = DataIterator(indexes, batch_size=self.batch_size, shuffle=True, drop_last=True)  # shuffle batch index
            for bat_index in indexes_iter:
                yield bat_index

    def _build_session_graph(self, bat_items):
        A_in, A_out, alias_inputs = [], [], []
        all_mask = [[1] * len(items) for items in bat_items]
        bat_items = pad_sequences(bat_items, value=self.num_item)

        unique_nodes = [np.unique(items).tolist() for items in bat_items]
        max_n_node = np.max([len(nodes) for nodes in unique_nodes])
        for u_seq, u_node, mask in zip(bat_items, unique_nodes, all_mask):
            adj_mat = np.zeros((max_n_node, max_n_node))
            id_map = {node: idx for idx, node in enumerate(u_node)}
            if len(u_seq) > 1:
                alias_previous = [id_map[i] for i in u_seq[:len(mask) - 1]]
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

        items = pad_sequences(unique_nodes, value=self.num_item)
        all_mask = pad_sequences(all_mask, value=0)
        return A_in, A_out, alias_inputs, items, all_mask

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, items):
        users = DataIterator(users, batch_size=self.batch_size, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            cur_batch_size = len(bat_user)
            bat_items = [self.user_pos_train[user][-self.max_seq_len:] for user in bat_user]
            bat_adj_in, bat_adj_out, bat_alias, bat_items, bat_mask = self._build_session_graph(bat_items)
            if cur_batch_size < self.batch_size:  # padding
                pad_size = self.batch_size - cur_batch_size
                bat_adj_in = np.concatenate([bat_adj_in, [bat_adj_in[-1]] * pad_size], axis=0)
                bat_adj_out = np.concatenate([bat_adj_out, [bat_adj_out[-1]] * pad_size], axis=0)
                bat_alias = np.concatenate([bat_alias, [bat_alias[-1]] * pad_size], axis=0)
                bat_items = np.concatenate([bat_items, [bat_items[-1]] * pad_size], axis=0)
                bat_mask = np.concatenate([bat_mask, [bat_mask[-1]] * pad_size], axis=0)

            feed = {self.item_ph: bat_items,
                    self.adj_in_ph: bat_adj_in,
                    self.adj_out_ph: bat_adj_out,
                    self.alias_ph: bat_alias,
                    self.mask_ph: bat_mask}
            bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
            all_ratings.extend(bat_ratings[:cur_batch_size])
        all_ratings = np.array(all_ratings)
        if items is not None:
            all_ratings = [all_ratings[idx][u_item] for idx, u_item in enumerate(items)]

        return all_ratings
