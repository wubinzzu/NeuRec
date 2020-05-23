"""
Reference: Xiangnan He et al., "NAIS: Neural Attentive Item Similarity Model for Recommendation." in TKDE2018
@author: wubin
"""
from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner,data_generator, tool
from util import timer
from util.tool import csr_to_user_dict
import pickle
from util import l2_loss
from util import pad_sequences
from util.data_iterator import DataIterator


class NAIS(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(NAIS, self).__init__(dataset, conf)
        self.pretrain = conf["pretrain"]
        self.verbose = conf["verbose"]
        self.batch_size = conf["batch_size"]
        self.num_epochs = conf["epochs"]
        self.weight_size = conf["weight_size"]
        self.embedding_size = conf["embedding_size"]
        self.data_alpha = conf["data_alpha"]
        self.regs = conf["regs"]
        self.is_pairwise = conf["is_pairwise"]
        self.topK = conf["topk"]
        self.lambda_bilinear = self.regs[0]
        self.gamma_bilinear = self.regs[1]
        self.eta_bilinear = self.regs[2] 
        self.alpha = conf["alpha"]
        self.beta = conf["beta"]
        self.num_negatives = conf["num_neg"]
        self.learning_rate = conf["learning_rate"]
        self.activation = conf["activation"]
        self.loss_function = conf["loss_function"]
        self.algorithm = conf["algorithm"]
        self.learner = conf["learner"]
        self.embed_init_method = conf["embed_init_method"]
        self.weight_init_method = conf["weight_init_method"]
        self.stddev = conf["stddev"]
        self.pretrain_file = conf["pretrain_file"]
        self.dataset = dataset
        self.num_items = dataset.num_items
        self.num_users = dataset.num_users
        self.train_dict = csr_to_user_dict(self.dataset.train_matrix)
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None], name="user_input")  # the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None], name="num_idx")  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None], name="item_input_pos")  # the index of items
            if self.is_pairwise is True:
                self.user_input_neg = tf.placeholder(tf.int32, shape=[None, None], name="user_input_neg")
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")
                self.num_idx_neg = tf.placeholder(tf.float32, shape=[None], name="num_idx_neg")
            else:
                self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")

    def _create_variables(self, params=None):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            if params is None:
                embed_initializer = tool.get_initializer(self.embed_init_method, self.stddev)
                
                self.c1 = tf.Variable(embed_initializer([self.num_items, self.embedding_size]),
                                      name='c1', dtype=tf.float32)
                self.embedding_Q = tf.Variable(embed_initializer([self.num_items, self.embedding_size]),
                                               name='embedding_Q', dtype=tf.float32)
                self.bias = tf.Variable(tf.zeros(self.num_items), name='bias')
            else:
                self.c1 = tf.Variable(params[0], name='c1', dtype=tf.float32)
                self.embedding_Q = tf.Variable(params[1], name='embedding_Q', dtype=tf.float32)
                self.bias = tf.Variable(params[2], name="bias", dtype=tf.float32)
                
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], axis=0, name='embedding_Q_')

            # Variables for attention
            weight_initializer = tool.get_initializer(self.weight_init_method, self.stddev)
            if self.algorithm == 0:
                self.W = tf.Variable(weight_initializer([self.embedding_size, self.weight_size]),
                                     name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            else:    
                self.W = tf.Variable(weight_initializer([2*self.embedding_size, self.weight_size]),
                                     name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            
            self.b = tf.Variable(weight_initializer([1, self.weight_size]),
                                 name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)
            
    def _create_inference(self, user_input, item_input, num_idx):
        with tf.name_scope("inference"):
            embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, user_input)  # (b, n, e)
            embedding_q = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)  # (b, 1, e)
            
            if self.algorithm == 0:
                embedding_p = self._attention_mlp(embedding_q_ * embedding_q, embedding_q_, num_idx)
            else:
                n = tf.shape(user_input)[1]
                embedding_p = self._attention_mlp(tf.concat([embedding_q_, tf.tile(embedding_q, tf.stack([1, n, 1]))], 2),
                                                  embedding_q_, num_idx)

            embedding_q = tf.reduce_sum(embedding_q, 1)
            bias_i = tf.nn.embedding_lookup(self.bias, item_input)
            coeff = tf.pow(num_idx, tf.constant(self.alpha, tf.float32, [1]))
            output = coeff * tf.reduce_sum(embedding_p*embedding_q, 1) + bias_i
            
            return embedding_q_, embedding_q, output
    
    def _create_loss(self):
        with tf.name_scope("loss"):
            p1, q1, self.output = self._create_inference(self.user_input,self.item_input,self.num_idx)
            if self.is_pairwise is True:
                _, q2, output_neg = self._create_inference(self.user_input_neg, self.item_input_neg, self.num_idx_neg)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function, self.result) + \
                            self.lambda_bilinear * l2_loss(p1) + \
                            self.gamma_bilinear * l2_loss(q2, q1)
            
            else:
                self.loss = learner.pointwise_loss(self.loss_function, self.labels, self.output) + \
                            self.lambda_bilinear * l2_loss(p1) + \
                            self.gamma_bilinear * l2_loss(q1)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
            
    def build_graph(self):
        self._create_placeholders()
        try:
            pre_trained_params = []
            with open(self.pretrain_file, "rb") as fin:
                pre_trained_params.append(pickle.load(fin, encoding="utf-8"))
            with open(self.mlp_pretrain, "rb") as fin:
                pre_trained_params.append(pickle.load(fin, encoding="utf-8"))
            self.logger.info("load pretrained params successful!")
        except:
            pre_trained_params = None
            self.logger.info("load pretrained params unsuccessful!")
            
        self._create_variables(pre_trained_params)
        self._create_loss()
        self._create_optimizer()

    def _attention_mlp(self, q_, embedding_q_, num_idx):
            with tf.name_scope("attention_MLP"):
                b = tf.shape(q_)[0]
                n = tf.shape(q_)[1]
                r = (self.algorithm + 1)*self.embedding_size

                mlp_output = tf.matmul(tf.reshape(q_, [-1, r]), self.W) + self.b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
                if self.activation == 0:
                    mlp_output = tf.nn.relu(mlp_output)
                elif self.activation == 1:
                    mlp_output = tf.nn.sigmoid(mlp_output)
                elif self.activation == 2:
                    mlp_output = tf.nn.tanh(mlp_output)
    
                A_ = tf.reshape(tf.matmul(mlp_output, self.h), [b,n])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)
    
                # softmax for not mask features
                exp_A_ = tf.exp(A_)
                mask_mat = tf.sequence_mask(num_idx, maxlen = n, dtype=tf.float32)  # (b, n)
                exp_A_ = mask_mat * exp_A_
                exp_sum = tf.reduce_sum(exp_A_, 1, keepdims=True)  # (b, 1)
                exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))
    
                A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)
    
                return tf.reduce_sum(A * embedding_q_, 1)

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(1, self.num_epochs+1):
            if self.is_pairwise is True:
                user_input, user_input_neg, num_idx_pos, num_idx_neg, item_input_pos, item_input_neg = \
                    data_generator._get_pairwise_all_likefism_data(self.dataset)
                data_iter = DataIterator(user_input, user_input_neg, num_idx_pos,
                                         num_idx_neg, item_input_pos, item_input_neg,
                                         batch_size=self.batch_size, shuffle=True)
            else:
                user_input, num_idx, item_input, labels = \
                 data_generator._get_pointwise_all_likefism_data(self.dataset, self.num_negatives, self.train_dict)
                data_iter = DataIterator(user_input, num_idx, item_input, labels,
                                         batch_size=self.batch_size, shuffle=True)
           
            num_training_instances = len(user_input)
            total_loss = 0.0
            training_start_time = time()
            if self.is_pairwise is True:
                for bat_users_pos, bat_users_neg, bat_idx_pos, bat_idx_neg, bat_items_pos, bat_items_neg in data_iter:
                    bat_users_pos = pad_sequences(bat_users_pos, value=self.num_items)
                    bat_users_neg = pad_sequences(bat_users_neg, value=self.num_items)
                    feed_dict = {self.user_input: bat_users_pos,
                                 self.user_input_neg: bat_users_neg,
                                 self.num_idx: bat_idx_pos,
                                 self.num_idx_neg: bat_idx_neg,
                                 self.item_input: bat_items_pos,
                                 self.item_input_neg: bat_items_neg}

                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
            else:
                for bat_users, bat_idx, bat_items, bat_labels in data_iter:
                    bat_users = pad_sequences(bat_users, value=self.num_items)
                    feed_dict = {self.user_input: bat_users,
                                 self.num_idx: bat_idx,
                                 self.item_input: bat_items,
                                 self.labels: bat_labels}
                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss

            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/num_training_instances,
                                                             time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
        
        # save model
        # params = self.sess.run([self.c1, self.embedding_Q, self.bias])
        # with open("./pretrained/%s_epoch=%d_fism.pkl" % (self.dataset_name, self.num_epochs), "wb") as fout:
        #     pickle.dump(params, fout)
    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)
    
    def predict(self, user_ids, candidate_items_userids):      
        ratings = []
        if candidate_items_userids is not None:
            for u, eval_items_by_u in zip(user_ids, candidate_items_userids):
                user_input = []
                cand_items_by_u = self.train_dict[u]
                num_idx = len(cand_items_by_u)
                item_idx = np.full(len(eval_items_by_u), num_idx, dtype=np.int32)
                user_input.extend([cand_items_by_u]*len(eval_items_by_u))
                feed_dict = {self.user_input: user_input,
                             self.num_idx: item_idx, 
                             self.item_input: eval_items_by_u}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
                
        else:
            eval_items = np.arange(self.num_items)
            for u in user_ids:
                user_input = []
                cand_items_by_u = self.train_dict[u]
                num_idx = len(cand_items_by_u)
                item_idx = np.full(self.num_items, num_idx, dtype=np.int32)
                user_input.extend([cand_items_by_u]*self.num_items)
                feed_dict = {self.user_input: user_input,
                             self.num_idx: item_idx, 
                             self.item_input: eval_items}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        return ratings
