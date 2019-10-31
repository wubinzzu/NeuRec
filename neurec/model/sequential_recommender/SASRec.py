"""
Paper: Self-Attentive Sequential Recommendation
Author: Wang-Cheng Kang, and Julian McAuley
Reference: https://github.com/kang205/SASRec
@author: Zhongchuan Sun
"""

import numpy as np
from neurec.model.AbstractRecommender import SeqAbstractRecommender
from neurec.util import DataIterator
from neurec.util.tool import csr_to_user_dict_bytime
import tensorflow as tf
from neurec.util.tool import inner_product
from neurec.util.tool import batch_random_choice
from neurec.util.tool import pad_sequences

def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    """Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0, is_training=True,
                        causality=False, scope="multihead_attention", reuse=None, with_qk=False):
    """Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

    if with_qk:
        return Q, K
    else:
        return outputs


def feedforward(inputs, num_units, scope="multihead_attention",
                dropout_rate=0.2, is_training=True, reuse=None):
    """Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

    return outputs


class SASRec(SeqAbstractRecommender):
    properties = [
        "lr",
        "l2_emb",
        "hidden_units",
        "batch_size",
        "epochs",
        "dropout",
        "max_len",
        "num_blocks",
        "num_heads"
    ]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        train_matrix, time_matrix = self.dataset.train_matrix, self.dataset.time_matrix
        self.users_num, self.items_num = train_matrix.shape

        self.lr = self.conf["lr"]
        self.l2_emb = self.conf["l2_emb"]
        self.hidden_units = self.conf["hidden_units"]
        self.batch_size = self.conf["batch_size"]
        self.epochs = self.conf["epochs"]
        self.dropout_rate = self.conf["dropout"]
        self.max_len = self.conf["max_len"]
        self.num_blocks = self.conf["num_blocks"]
        self.num_heads = self.conf["num_heads"]

        self.user_pos_train = csr_to_user_dict_bytime(train_matrix, time_matrix)

        

    def _create_variable(self):
        # self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.item_seq_ph = tf.placeholder(tf.int32, [None, self.max_len], name="item_seq")
        self.item_pos_ph = tf.placeholder(tf.int32, [None, self.max_len], name="item_pos")
        self.item_neg_ph = tf.placeholder(tf.int32, [None, self.max_len], name="item_neg")
        self.is_training = tf.placeholder(tf.bool, name="training_flag")
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.l2_emb)
        item_embeddings = tf.get_variable('item_embeddings', dtype=tf.float32,
                                          shape=[self.items_num, self.hidden_units],
                                          regularizer=l2_regularizer)

        zero_pad = tf.zeros([1, self.hidden_units], name="padding")
        self.item_embeddings = tf.concat([item_embeddings, zero_pad], axis=0)

        self.position_embeddings = tf.get_variable('position_embeddings', dtype=tf.float32,
                                                   shape=[self.max_len, self.hidden_units],
                                                   regularizer=l2_regularizer)

    def _build_model(self):
        self._create_variable()
        # embedding layer
        batch_size = tf.shape(self.item_seq_ph)[0]
        item_seq_emb = tf.nn.embedding_lookup(self.item_embeddings, self.item_seq_ph)

        position = tf.tile(tf.expand_dims(tf.range(self.max_len), 0), [batch_size, 1])
        position_emb = tf.nn.embedding_lookup(self.position_embeddings, position)

        item_seq_emb = item_seq_emb + position_emb

        # dropout and mask
        item_seq_emb = tf.layers.dropout(item_seq_emb, rate=self.dropout_rate, training=self.is_training)
        # the id of padding items is 'items_num'
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.item_seq_ph, self.items_num)), -1)  # (b, l, 1)
        item_seq_emb = tf.multiply(item_seq_emb, mask)

        # build blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                item_seq_emb = multihead_attention(queries=normalize(item_seq_emb),
                                                   keys=item_seq_emb,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                # feed forward
                item_seq_emb = feedforward(normalize(item_seq_emb), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training)

                item_seq_emb = tf.multiply(item_seq_emb, mask)

        item_seq_emb = normalize(item_seq_emb)  # (b, l, d)

        item_pos = tf.reshape(self.item_pos_ph, [batch_size * self.max_len])  # (b*l,)
        item_neg = tf.reshape(self.item_neg_ph, [batch_size * self.max_len])  # (b*l,)

        pos_emb = tf.nn.embedding_lookup(self.item_embeddings, item_pos)  # (b*l, d)
        neg_emb = tf.nn.embedding_lookup(self.item_embeddings, item_neg)  # (b*l, d)
        seq_emb = tf.reshape(item_seq_emb, [batch_size * self.max_len, self.hidden_units])  # (b*l, d)

        # rating calculation
        pos_logits = inner_product(pos_emb, seq_emb)  # (b*l,)
        neg_logits = inner_product(neg_emb, seq_emb)  # (b*l,)

        # loss calculation
        # ignore padding items (0)
        target_mask = tf.to_float(tf.not_equal(self.item_pos_ph, self.items_num))  # (b, l)
        target_mask = tf.reshape(target_mask, [batch_size*self.max_len])  # (b*l,)

        pos_loss = -tf.log(tf.sigmoid(pos_logits) + 1e-24) * target_mask
        neg_loss = -tf.log(1 - tf.sigmoid(neg_logits) + 1e-24) * target_mask
        loss = tf.reduce_sum(pos_loss+neg_loss) / tf.reduce_sum(target_mask)
        try:
            reg_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss = loss + reg_losses
        except:
            pass

        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta2=0.98).minimize(loss)

        # for predication/test
        items_embeddings = self.item_embeddings[:-1]  # remove the padding item
        last_emb = item_seq_emb[:, -1, :]  # (b, d), the embedding of last item of each session
        self.all_logits = tf.matmul(last_emb, items_embeddings, transpose_b=True)

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())

        for epoch in range(self.epochs):
            item_seq_list, item_pos_list, item_neg_list = self.get_train_data()
            data = DataIterator(item_seq_list, item_pos_list, item_neg_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_item_seq, bat_item_pos, bat_item_neg in data:
                feed = {self.item_seq_ph: bat_item_seq,
                        self.item_pos_ph: bat_item_pos,
                        self.item_neg_ph: bat_item_neg,
                        self.is_training: True}

                self.sess.run(self.train_opt, feed_dict=feed)

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def get_train_data(self):
        item_seq_list, item_pos_list, item_neg_list = [], [], []
        all_users = DataIterator(list(self.user_pos_train.keys()), batch_size=1024, shuffle=False)
        for bat_users in all_users:
            bat_seq = [self.user_pos_train[u][:-1] for u in bat_users]
            bat_pos = [self.user_pos_train[u][1:] for u in bat_users]
            n_neg_items = [len(pos) for pos in bat_pos]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_random_choice(self.items_num, n_neg_items, replace=True, exclusion=exclusion)

            # padding
            bat_seq = pad_sequences(bat_seq, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')
            bat_pos = pad_sequences(bat_pos, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')
            bat_neg = pad_sequences(bat_neg, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')

            item_seq_list.extend(bat_seq)
            item_pos_list.extend(bat_pos)
            item_neg_list.extend(bat_neg)
        return item_seq_list, item_pos_list, item_neg_list

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, items=None):
        users = DataIterator(users, batch_size=512, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            bat_seq = [self.user_pos_train[u] for u in bat_user]
            bat_seq = pad_sequences(bat_seq, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')
            feed = {self.item_seq_ph: bat_seq,
                    self.is_training: False}
            bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
            all_ratings.extend(bat_ratings)
        all_ratings = np.array(all_ratings, dtype=np.float32)
        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]
        return all_ratings
