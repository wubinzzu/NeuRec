'''
Reference: Wang Xiang et al. "Neural Graph Collaborative Filtering." in SIGIR2019
Source code: https://github.com/xiangwang1223/neural_graph_collaborative_filtering
@author: wubin
'''
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from time import time
from neurec.util import data_gen, reader
from neurec.evaluation import Evaluate
from neurec.model.AbstractRecommender import AbstractRecommender
class NGCF(AbstractRecommender):
    def __init__(self,sess,dataset):
        self.conf = reader.config("NGCF.properties", "hyperparameters")

        print("NGCF arguments: %s " %(self.conf))
        self.learning_rate = float(self.conf["learning_rate"])
        self.learner = self.conf["learner"]
        self.topK = int(self.conf["topk"])
        self.batch_size= int(self.conf["batch_size"])
        self.emb_dim = int(self.conf["embedding_size"])
        self.weight_size = eval(self.conf["layer_size"])
        self.n_layers = len(self.weight_size)
        self.num_epochs=int(self.conf["epochs"])
        self.decay=float(self.conf["reg"])
        self.loss_function=self.conf["loss_function"]
        self.node_dropout_flag = self.conf["node_dropout_flag"]
        self.adj_type = self.conf["adj_type"]
        self.alg_type = self.conf["alg_type"]
        self.n_fold = 100
        self.verbose=int(self.conf["verbose"])
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items    
        self.graph = dataset.trainMatrix.toarray()
        self.norm_adj= self.get_adj_mat()
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.pretrain_data = None
        self.sess=sess        

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            # placeholder definition
            self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
            self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
            self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
    
            # dropout: node dropout (adopted on the ego-networks);
            #          ... since the usage of node dropout have higher computational cost,
            #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
            #          message dropout (adopted on the convolution operations).
            self.node_dropout = tf.compat.v1.placeholder_with_default([0.], shape=[None])
            self.mess_dropout = tf.compat.v1.placeholder_with_default([0.], shape=[None])
            
    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now   
            """
            *********************************************************
            Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
            Different Convolutional Layers:
                1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
                2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
                3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
            """
            # initialization of model parameters
            self.weights = self._init_weights()
            
            if self.alg_type in ['ngcf']:
                self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
    
            elif self.alg_type in ['gcn']:
                self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()
    
            elif self.alg_type in ['gcmc']:
                self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()
    
    
    def _create_inference(self):
        with tf.name_scope("inference"):
            """
            *********************************************************
            Establish the final representations for user-item pairs in batch.
            """
            self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
            self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
            self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
    
            """
            *********************************************************
            Inference for the testing phase.
            """
            self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

            
    def _create_loss(self):
        with tf.name_scope("loss"): 
            
            self.pos_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings), axis=1)
            neg_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.neg_i_g_embeddings), axis=1)
    
            regularizer = tf.nn.l2_loss(self.u_g_embeddings) + tf.nn.l2_loss(self.pos_i_g_embeddings) + tf.nn.l2_loss(self.neg_i_g_embeddings)
            regularizer = regularizer/self.batch_size
    
            maxi = tf.log(tf.nn.sigmoid(self.pos_scores - neg_scores))
            mf_loss = tf.negative(tf.reduce_mean(maxi))
    
            emb_loss = self.decay * regularizer
    
            reg_loss = tf.constant(0.0, tf.float32, [1])
            
            self.loss = mf_loss + emb_loss + reg_loss
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        
    def train_model(self):
        for epoch in  range(self.num_epochs):
            # Generate training instances
            user_input, item_input_pos, item_input_neg = data_gen._get_pairwise_all_data(self.dataset)
            
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                bat_users,bat_items_pos,bat_items_neg =\
                 data_gen._get_pairwise_batch_data(user_input,\
                 item_input_pos, item_input_neg, num_batch, self.batch_size)
                feed_dict = {self.users: bat_users, self.pos_items: bat_items_pos,
                              self.node_dropout: [0.1],
                              self.mess_dropout: [0.1],
                              self.neg_items: bat_items_neg}
                      
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
            
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)
                
    def predict(self, user_id, items):
        users = np.full(len(items), user_id, dtype=np.int32)
        return self.sess.run(self.pos_scores, feed_dict={self.users: users, self.pos_items: items})
    
    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag =='True':
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)


        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        return u_g_embeddings, i_g_embeddings
        
    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.num_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.num_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights
     
    def normalized_adj_single(self,adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()
    def get_adj_mat(self):
        A = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        A = A.tolil()
        A[:self.num_users, self.num_users:] = self.graph
        A[self.num_users:, :self.num_users] = self.graph.transpose()
        A = A.todok()
        if self.adj_type== 'plain':
            adj_mat = A
            print('use the plain adjacency matrix')
        elif self.adj_type == 'norm':  
            adj_mat = self.normalized_adj_single(A + sp.eye(A.shape[0]))
            print('use the normalized adjacency matrix')
        elif self.adj_type == 'gcmc':
            adj_mat = self.normalized_adj_single(A)
            print('use the gcmc adjacency matrix')
        else:
            adj_mat = self.normalized_adj_single(A) + sp.eye(A.shape[0])
            print('use the mean adjacency matrix')
    
        return adj_mat.tocsr()
    
    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))
        return A_fold_hat
    
    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)