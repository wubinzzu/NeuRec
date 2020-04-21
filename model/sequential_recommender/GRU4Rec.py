"""
Paper: Session-based Recommendations with Recurrent Neural Networks
Author: Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk
Reference: https://github.com/hidasib/GRU4Rec
           https://github.com/Songweiping/GRU4Rec_TensorFlow
@author: Zhongchuan Sun
"""

import numpy as np
from model.AbstractRecommender import SeqAbstractRecommender
import tensorflow as tf
from util import log_loss, l2_loss


class GRU4Rec(SeqAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(GRU4Rec, self).__init__(dataset, conf)
        self.train_matrix = dataset.train_matrix
        self.dataset = dataset

        self.users_num, self.items_num = self.train_matrix.shape

        self.lr = conf["lr"]
        self.reg = conf["reg"]
        self.layers = conf["layers"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]

        if conf["hidden_act"] == "relu":
            self.hidden_act = tf.nn.relu
        elif conf["hidden_act"] == "tanh":
            self.hidden_act = tf.nn.tanh
        else:
            raise ValueError("There is not hidden_act named '%s'." % conf["hidden_act"])

        # final_act = leaky-relu
        if conf["final_act"] == "relu":
            self.final_act = tf.nn.relu
        elif conf["final_act"] == "linear":
            self.final_act = tf.identity
        elif conf["final_act"] == "leaky_relu":
            self.final_act = tf.nn.leaky_relu
        else:
            raise ValueError("There is not final_act named '%s'." % conf["final_act"])

        if conf["loss"] == "bpr":
            self.loss_fun = self._bpr_loss
        elif conf["loss"] == "top1":
            self.loss_fun = self._top1_loss
        else:
            raise ValueError("There is not loss named '%s'." % conf["loss"])

        self.data_uit, self.offset_idx = self._init_data()

        # for sampling negative items
        _, pop = np.unique(self.data_uit[:, 1], return_counts=True)
        pop_cumsum = np.cumsum(pop)
        self.pop_cumsum = pop_cumsum / pop_cumsum[-1]

        self.sess = sess

    def _init_data(self):
        time_dok = self.dataset.time_matrix.todok()
        data_uit = [[row, col, time] for (row, col), time in time_dok.items()]
        data_uit.sort(key=lambda x: (x[0], x[-1]))
        data_uit = np.array(data_uit, dtype=np.int32)
        _, idx = np.unique(data_uit[:, 0], return_index=True)
        offset_idx = np.zeros(len(idx)+1, dtype=np.int32)
        offset_idx[:-1] = idx
        offset_idx[-1] = len(data_uit)

        return data_uit, offset_idx

    def _create_variable(self):
        self.X_ph = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y_ph = tf.placeholder(tf.int32, [self.batch_size], name='output')
        self.state_ph = [tf.placeholder(tf.float32, [self.batch_size, n_unit], name='layer_%d_state' % idx)
                         for idx, n_unit in enumerate(self.layers)]

        init = tf.random.truncated_normal([self.items_num, self.layers[0]], mean=0.0, stddev=0.01)
        self.input_embeddings = tf.Variable(init, dtype=tf.float32, name="input_embeddings")

        init = tf.random.truncated_normal([self.items_num, self.layers[-1]], mean=0.0, stddev=0.01)
        self.item_embeddings = tf.Variable(init, dtype=tf.float32, name="item_embeddings")
        self.item_biases = tf.Variable(tf.zeros([self.items_num]), dtype=tf.float32, name="item_biases")

    def _bpr_loss(self, logits):
        # logits: (b, size_y)
        pos_logits = tf.matrix_diag_part(logits)  # (b,)
        pos_logits = tf.reshape(pos_logits, shape=[-1, 1])  # (b, 1)
        loss = tf.reduce_mean(log_loss(pos_logits-logits))
        return loss

    def _top1_loss(self, logits):
        # logits: (b, size_y)
        pos_logits = tf.matrix_diag_part(logits)  # (b,)
        pos_logits = tf.reshape(pos_logits, shape=[-1, 1])  # (b, 1)
        loss1 = tf.reduce_mean(tf.sigmoid(-pos_logits + logits), axis=-1)  # (b,)
        loss2 = tf.reduce_mean(tf.sigmoid(tf.pow(logits, 2)), axis=-1) - \
                tf.squeeze(tf.sigmoid(tf.pow(pos_logits, 2))/self.batch_size)  # (b,)
        return tf.reduce_mean(loss1+loss2)

    def build_graph(self):
        self._create_variable()
        # get embedding and bias
        # b: batch size
        # l1: the dim of the first layer
        # ln: the dim of the last layer
        # size_y: the length of Y_ph, i.e., n_sample+batch_size

        cells = [tf.nn.rnn_cell.GRUCell(size, activation=self.hidden_act) for size in self.layers]
        drop_cell = [tf.nn.rnn_cell.DropoutWrapper(cell) for cell in cells]
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(drop_cell)
        inputs = tf.nn.embedding_lookup(self.input_embeddings, self.X_ph)  # (b, l1)
        outputs, state = stacked_cell(inputs, state=self.state_ph)
        self.u_emb = outputs  # outputs: (b, ln)
        self.final_state = state  # [(b, l1), (b, l2), ..., (b, ln)]

        # for training
        items_embed = tf.nn.embedding_lookup(self.item_embeddings, self.Y_ph)  # (size_y, ln)
        items_bias = tf.gather(self.item_biases, self.Y_ph)  # (size_y,)

        logits = tf.matmul(outputs, items_embed, transpose_b=True) + items_bias  # (b, size_y)
        logits = self.final_act(logits)

        loss = self.loss_fun(logits)

        # reg loss

        reg_loss = l2_loss(inputs, items_embed, items_bias)
        final_loss = loss + self.reg*reg_loss
        self.update_opt = tf.train.AdamOptimizer(self.lr).minimize(final_loss)

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())

        data_uit, offset_idx = self.data_uit, self.offset_idx
        data_items = data_uit[:, 1]

        for epoch in range(self.epochs):
            state = [np.zeros([self.batch_size, n_unit], dtype=np.float32) for n_unit in self.layers]
            user_idx = np.random.permutation(len(offset_idx) - 1)
            iters = np.arange(self.batch_size, dtype=np.int32)
            maxiter = iters.max()
            start = offset_idx[user_idx[iters]]
            end = offset_idx[user_idx[iters]+1]
            finished = False
            while not finished:
                min_len = (end - start).min()
                out_idx = data_items[start]
                for i in range(min_len-1):
                    in_idx = out_idx
                    out_idx = data_items[start+i+1]
                    out_items = out_idx

                    feed = {self.X_ph: in_idx, self.Y_ph: out_items}
                    for l in range(len(self.layers)):
                        feed[self.state_ph[l]] = state[l]

                    _, state = self.sess.run([self.update_opt, self.final_state], feed_dict=feed)

                start = start+min_len-1
                mask = np.arange(len(iters))[(end - start) <= 1]
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_idx)-1:
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = offset_idx[user_idx[maxiter]]
                    end[idx] = offset_idx[user_idx[maxiter]+1]
                if len(mask):
                    for i in range(len(self.layers)):
                        state[i][mask] = 0

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _get_user_embeddings(self):
        users = np.arange(self.users_num, dtype=np.int32)
        u_nnz = np.array([self.train_matrix[u].nnz for u in users], dtype=np.int32)
        users = users[np.argsort(-u_nnz)]
        user_embeddings = np.zeros([self.users_num, self.layers[-1]], dtype=np.float32)  # saving user embedding

        data_uit, offset_idx = self.data_uit, self.offset_idx
        data_items = data_uit[:, 1]

        state = [np.zeros([self.batch_size, n_unit], dtype=np.float32) for n_unit in self.layers]
        batch_iter = np.arange(self.batch_size, dtype=np.int32)
        next_iter = batch_iter.max() + 1

        start = offset_idx[users[batch_iter]]
        end = offset_idx[users[batch_iter] + 1]  # the start index of next user

        batch_mask = np.ones([self.batch_size], dtype=np.int32)
        while np.sum(batch_mask) > 0:
            min_len = (end - start).min()

            for i in range(min_len):
                cur_items = data_items[start + i]
                feed = {self.X_ph: cur_items}
                for l in range(len(self.layers)):
                    feed[self.state_ph[l]] = state[l]

                u_emb, state = self.sess.run([self.u_emb, self.final_state], feed_dict=feed)

            start = start + min_len
            mask = np.arange(self.batch_size)[(end - start) == 0]
            for idx in mask:
                u = users[batch_iter[idx]]
                user_embeddings[u] = u_emb[idx]  # saving user embedding
                if next_iter < self.users_num:
                    batch_iter[idx] = next_iter
                    start[idx] = offset_idx[users[next_iter]]
                    end[idx] = offset_idx[users[next_iter] + 1]
                    next_iter += 1
                else:
                    batch_mask[idx] = 0
                    start[idx] = 0
                    end[idx] = offset_idx[-1]

            for i, _ in enumerate(self.layers):
                state[i][mask] = 0

        return user_embeddings

    def evaluate_model(self):
        self.cur_user_embeddings = self._get_user_embeddings()
        self.cur_item_embeddings, self.cur_item_biases = self.sess.run([self.item_embeddings, self.item_biases])
        return self.evaluator.evaluate(self)

    def predict(self, users, items=None):
        user_embeddings = self.cur_user_embeddings[users]
        all_ratings = np.matmul(user_embeddings, self.cur_item_embeddings.T) + self.cur_item_biases

        # final_act = leaky-relu
        if self.final_act == tf.nn.relu:
            all_ratings = np.maximum(all_ratings, 0)
        elif self.final_act == tf.identity:
            all_ratings = all_ratings
        elif self.final_act == tf.nn.leaky_relu:
            all_ratings = np.maximum(all_ratings, all_ratings*0.2)
        else:
            pass

        all_ratings = np.array(all_ratings, dtype=np.float32)
        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]

        return all_ratings
