from __future__ import absolute_import
from __future__ import division

import os
from model.AbstractRecommender import AbstractRecommender
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import logging
from time import time
from evaluation import Evaluate
import configparser
from util import learner
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
class DeepICF(AbstractRecommender):
    def __init__(self,sess,dataset):
        config = configparser.ConfigParser()
        config.read("conf/DeepICF.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("DeepICF arguments: %s " %(self.conf)) 
        self.pretrain = int(self.conf["pretrain"])
        self.verbose = int(self.conf["verbose"])
        self.batch_choice = str(self.conf["batch_choice"])
        self.batch_size = int(self.conf["batch_size"]) 
        self.use_batch_norm = int(self.conf["batch_norm"])    
        self.num_epochs = int(self.conf["epochs"])
        self.weight_size = int(self.conf["weight_size"])
        self.embedding_size = int(self.conf["embedding_size"])
        self.n_hidden = eval(self.conf["layers"])
        self.regs = eval(self.conf["regs"])
        self.reg_W = eval(self.conf["regw"])
        self.ispairwise = self.conf["ispairwise"]
        self.topK = int(self.conf["topk"])
        self.lambda_bilinear = self.regs[0]
        self.gamma_bilinear = self.regs[1]
        self.eta_bilinear = self.regs[2] 
        self.alpha = float(self.conf["alpha"]) 
        self.beta = float(self.conf["beta"])
        self.num_negatives= int(self.conf["num_neg"])
        self.learning_rate = float(self.conf["learning_rate"])
        self.activation = self.conf["activation"]
        self.loss_function = self.conf["loss_function"]
        self.algorithm = int(self.conf["algorithm"])
        self.learner = str(self.conf["learner"])
        self.dataset = dataset
        self.dataset_name= dataset.dataset_name 
        self.num_items = dataset.num_items
        self.num_users = dataset.num_users
        self.sess=sess  
    # batch norm
    def batch_norm_layer(self,x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
    
    
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])  # the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None,])  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None,])  # the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None,])  # the ground truth
            self.is_train_phase = tf.placeholder(tf.bool)  # mark is training or testing

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            trainable_flag = (self.pretrain != 2)
            self.c1 = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01), name='c1', dtype=tf.float32, trainable=trainable_flag)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], 0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01), name='embedding_Q', dtype=tf.float32, trainable=trainable_flag)
            self.bias = tf.Variable(tf.zeros(self.num_items), name='bias', trainable=trainable_flag)

            # Variables for attention
            if self.algorithm == 0:
                self.W = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.embedding_size))), name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            else:
                self.W = tf.Variable(tf.truncated_normal(shape=[2 * self.embedding_size, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + ( 2 * self.embedding_size)))), name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.embedding_size))), name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)

            # Variables for DeepICF+a
            self.weights = {
                'out': tf.Variable(tf.random_normal([self.n_hidden[-1], 1], mean=0, stddev=np.sqrt(2.0 / (self.n_hidden[-1] + 1))), name='weights_out')
            }
            self.biases = {
                'out': tf.Variable(tf.random_normal([1]), name='biases_out')
            }
            n_hidden_0 = self.embedding_size
            for i in range(len(self.n_hidden)):
                if i > 0:
                    n_hidden_0 = self.n_hidden[i - 1]
                n_hidden_1 = self.n_hidden[i]
                self.weights['h%d' % i] = tf.Variable(tf.random_normal([n_hidden_0, n_hidden_1], mean=0, stddev=np.sqrt(2.0 / (n_hidden_0 + n_hidden_1))), name='weights_h%d' % i)
                self.biases['b%d' % i] = tf.Variable(tf.random_normal([n_hidden_1]), name='biases_b%d' % i)


    def _attention_MLP(self, q_):
        with tf.name_scope("attention_MLP"):
            b = tf.shape(q_)[0]
            n = tf.shape(q_)[1]
            r = (self.algorithm + 1) * self.embedding_size

            MLP_output = tf.matmul(tf.reshape(q_, [-1, r]), self.W) + self.b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            if self.activation == 0:
                MLP_output = tf.nn.relu(MLP_output)
            elif self.activation == 1:
                MLP_output = tf.nn.sigmoid(MLP_output)
            elif self.activation == 2:
                MLP_output = tf.nn.tanh(MLP_output)

            A_ = tf.reshape(tf.matmul(MLP_output, self.h), [b, n])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            num_idx = self.num_idx
            mask_mat = tf.sequence_mask(num_idx, maxlen=n, dtype=tf.float32)  # (b, n)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keepdims=True)  # (b, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))

            A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)

            return A, tf.reduce_sum(A * self.embedding_q_, 1)

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, self.user_input)  # (b, n, e)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, self.item_input)  # (b, 1, e)

            if self.algorithm == 0:  # prod
                self.A, self.embedding_p = self._attention_MLP(self.embedding_q_ * tf.expand_dims(self.embedding_q,1))  # (?, k)
            else:  # concat
                n = tf.shape(self.user_input)[1]
                self.A, self.embedding_p = self._attention_MLP(tf.concat([self.embedding_q_, tf.tile(tf.expand_dims(self.embedding_q,1), tf.stack([1, n, 1]))], 2))  # (?, k)

            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            self.coeff = tf.pow(tf.expand_dims(self.num_idx,1), tf.constant(self.alpha, tf.float32, [1]))
            self.embedding_p = self.coeff * self.embedding_p  # (?, k)

            # DeepICF+a
            layer1 = tf.multiply(self.embedding_p, self.embedding_q)  # (?, k)
            for i in range(0,len(self.n_hidden)):
                layer1 = tf.add(tf.matmul(layer1, self.weights['h%d' % i]), self.biases['b%d' % i])
                if self.use_batch_norm:
                    layer1 = self.batch_norm_layer(layer1, train_phase=self.is_train_phase, scope_bn='bn_%d' % i)
                layer1 = tf.nn.relu(layer1)
            out_layer = tf.reduce_sum(tf.matmul(layer1, self.weights['out']) + self.biases['out'],1) # (?, 1)

            self.output = tf.sigmoid(tf.add_n([out_layer, self.bias_i]))  # (?, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                        self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + \
                        self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q_)) + \
                        self.eta_bilinear * tf.reduce_sum(tf.square(self.W))

            for i in range(min(len(self.n_hidden), len(self.reg_W))):
                if self.reg_W[i] > 0:
                    self.loss = self.loss + self.reg_W[i] * tf.reduce_sum(tf.square(self.weights['h%d'%i]))
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
            
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")
    def batch_gen(self,batches, i):  
        return [(batches[r])[i] for r in range(4)]    
    def train_model(self):
        algo = "DeepICF"
        log_dir = "Log/%s/" % self.dataset_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        filename = log_dir+"log_{}_model_{}_lr_reg{}.txt".\
        format(algo,self.dataset_name,self.learning_rate,self.lambda_bilinear)
        
        logging.basicConfig(filename=filename, level=logging.INFO)
        logging.info("begin training %s model ......" % algo)
        logging.info(self.conf)
        for epoch in  range(self.num_epochs):
            batches = self.shuffle()
            num_batch = len(batches[1])
            batch_index = np.arange(num_batch)
            training_start_time = time()
            total_loss = 0.0
            for index in batch_index:
                user_input, num_idx, item_input, labels = self.batch_gen(batches, index)
                feed_dict = {self.user_input: user_input, self.num_idx: num_idx, self.item_input: item_input,
                             self.labels: labels, self.is_train_phase: True}
                loss,_ = self.sess.run([self.loss, self.optimizer], feed_dict)
                total_loss+=loss
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_batch,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)

    def shuffle(self):   #negative sampling and shuffle the data
        if self.batch_choice == 'user':
            _user_input, _item_input, _labels, _batch_length = self._get_train_data_user()
            self._num_batch = len(_batch_length)
            user_input_list, num_idx_list, item_input_list, labels_list = [], [], [], []
            for i in range(int(self._num_batch)):
                ui, ni, ii, l = self._get_train_batch_user(i,_user_input, _item_input, _labels, _batch_length)
                user_input_list.append(ui)
                num_idx_list.append(ni)
                item_input_list.append(ii)
                labels_list.append(l)
            return user_input_list, num_idx_list, item_input_list, labels_list
        else:
            _user_input, _item_input, _labels = self._get_train_data_fixed()
            iterations = len(_user_input)
            self.index = np.arange(iterations)
            self._num_batch = iterations / self.batch_size
            user_input_list, num_idx_list, item_input_list, labels_list = [], [], [], []
            for i in range(int(self._num_batch)):
                ui, ni, ii, l = self._get_train_batch_fixed(i,_user_input, _item_input, _labels)
                user_input_list.append(ui)
                num_idx_list.append(ni)
                item_input_list.append(ii)
                labels_list.append(l)
            return user_input_list, num_idx_list, item_input_list, labels_list
    
    def _get_train_data_user(self):
        user_input, item_input, labels, batch_length = [],[],[],[]
        trainList = self.dataset.trainDict
        for u in range(self.num_users):
            if u == 0:
                batch_length.append((1+self.num_negatives) * len(trainList[u]))
            else:
                batch_length.append((1+self.num_negatives) * len(trainList[u])+batch_length[u-1])
            for i in trainList[u]:
                # positive instance
                user_input.append(u)
                item_input.append(i)
                labels.append(1)
                # negative instances
                for _ in range(self.num_negatives):
                    j = np.random.randint(self.num_items)
                    while j in trainList[u]:
                        j = np.random.randint(self.num_items)
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)
        return  user_input, item_input, labels, batch_length


    def _get_train_batch_user(self,i,user_input, item_input, labels, batch_length):
        #represent the feature of users via items rated by him/her
        user_list, num_list, item_list, labels_list = [],[],[],[]
        trainList = self.dataset.trainDict
        if i == 0:
            begin = 0
        else:
            begin = batch_length[i-1]
        batch_index = list(range(begin, batch_length[i]))
        np.random.shuffle(batch_index)
        for idx in batch_index:
            user_idx = user_input[idx]
            item_idx = item_input[idx]
            nonzero_row = []
            nonzero_row += trainList[user_idx]
            num_list.append(self._remove_item(self.num_items, nonzero_row, item_idx))
            user_list.append(nonzero_row)
            item_list.append(item_idx)
            labels_list.append(labels[idx])
        user_input = np.array(self._add_mask(self.num_items, user_list, max(num_list)))
        num_idx = np.array(num_list)
        item_input = np.array(item_list)
        labels = np.array(labels_list)
        return (user_input, num_idx, item_input, labels)

    def _get_train_data_fixed(self):
        user_input, item_input, labels = [],[],[]
        train = self.dataset.trainMatrix
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(self.num_items)
                while (u, j) in train.keys():
                # while train.has_key((u, j)):
                    j = np.random.randint(self.num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return user_input, item_input, labels

    def _get_train_batch_fixed(self,i,user_input, item_input, labels):
        #represent the feature of users via items rated by him/her
        user_list, num_list, item_list,labels_list = [],[],[],[]
        trainList = self.dataset.trainList
        begin = i * self.batch_size
        for idx in range(begin, begin+self.batch_size):
            user_idx = user_input[self.index[idx]]
            item_idx = item_input[self.index[idx]]
            nonzero_row = []
            nonzero_row += trainList[user_idx]
            num_list.append(self._remove_item(self.num_items, nonzero_row, item_idx))
            user_list.append(nonzero_row)
            item_list.append(item_idx)
            labels_list.append(labels[self.index[idx]])
        user_input = np.array(self._add_mask(self.num_items, user_list, max(num_list)))
        num_idx = np.array(num_list)
        item_input = np.array(item_list)
        labels = np.array(labels_list)
        return (user_input, num_idx, item_input, labels)

    def _remove_item(self,feature_mask, users, item):
        flag = 0
        for i in range(len(users)):
            if users[i] == item:
                users[i] = users[-1]
                users[-1] = feature_mask
                flag = 1
                break
        return len(users) - flag

    def _add_mask(self,feature_mask, features, num_max):
        #uniformalize the length of each batch
        for i in range(len(features)):
            features[i] = features[i] + [feature_mask] * (num_max+1 - len(features[i]))
        return features
    
    def predict(self, user_id,target_item,evaluateNegatives):
        items = [target_item]
        items.extend(evaluateNegatives)
        cand_items = self.dataset.trainDict[user_id]
        num_idx = len(cand_items)
        # Get prediction scores
        item_idx = np.full(len(items), num_idx, dtype=np.int32)
        user_input = []
        for _ in range(len(items)):
            user_input.append(cand_items)
        feed_dict = {self.user_input: np.array(user_input), \
                     self.num_idx: item_idx, self.item_input:items,self.is_train_phase: False}
        return self.sess.run((self.output), feed_dict=feed_dict)     