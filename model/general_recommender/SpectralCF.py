'''
Reference: Lei, Zheng, et al. "Spectral Collaborative Filtering." in RecSys2018
@author: wubin
'''
import tensorflow as tf
import numpy as np
from time import time
from util import DataGenerator, Learner, Tool
from util.Logger import logger
from model.AbstractRecommender import AbstractRecommender
from util.DataIterator import DataIterator
from util import timer
from util import l2_loss


class SpectralCF(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(SpectralCF, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.learner = conf["learner"]
        self.batch_size= conf["batch_size"]
        self.num_layers = conf["num_layers"]
        self.activation = conf["activation"]
        self.embedding_size = conf["embedding_size"]
        self.num_epochs = conf["epochs"]
        self.reg = conf["reg"]
        self.loss_function = conf["loss_function"]
        self.dropout = conf["dropout"]
        self.embed_init_method = conf["embed_init_method"]
        self.weight_init_method = conf["weight_init_method"]
        self.stddev = conf["stddev"]
        self.verbose = conf["verbose"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items  
        self.graph = dataset.train_matrix.toarray()
        self.A = self.adjacient_matrix(self_connection=True)
        self.D = self.degree_matrix()
        self.L = self.laplacian_matrix(normalized=True)
        self.lamda, self.U = np.linalg.eig(self.L)
        self.lamda = np.diag(self.lamda)
        self.sess=sess        

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.input_user = tf.placeholder(dtype=tf.int32, shape=[None,])
            self.input_item_pos = tf.placeholder(dtype=tf.int32, shape=[None,])
            self.input_item_neg = tf.placeholder(dtype=tf.int32, shape=[None,])
    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now  
            embed_initializer = Tool.get_initializer(self.embed_init_method, self.stddev)
            self.user_embeddings = tf.Variable(embed_initializer([self.num_users, self.embedding_size]),
                                     dtype=tf.float32, name='user_embeddings')
            self.item_embeddings = tf.Variable(embed_initializer([self.num_items, self.embedding_size]), 
                                     dtype=tf.float32, name='item_embeddings')
            
        weight_initializer = Tool.get_initializer(self.weight_init_method, self.stddev) 
        self.filters = []
        for _ in range(self.num_layers):
            self.filters.append(
                tf.Variable(weight_initializer([self.embedding_size, self.embedding_size]), dtype=tf.float32))
    
    def _create_inference(self):
        with tf.name_scope("inference"):
            A_hat = np.dot(self.U, self.U.T) + np.dot(np.dot(self.U, self.lamda), self.U.T)
            #A_hat += np.dot(np.dot(self.U, self.lamda_2), self.U.T)
            A_hat = A_hat.astype(np.float32)
    
            embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
            all_embeddings = [embeddings]
            for k in range(0, self.num_layers):
    
                embeddings = tf.matmul(A_hat, embeddings)
    
                #filters = self.filters[k]#tf.squeeze(tf.gather(self.filters, k))
                embeddings = Tool.activation_function(self.activation, (tf.matmul(embeddings, self.filters[k])))
                all_embeddings += [embeddings]
            all_embeddings = tf.concat(all_embeddings, 1)
            self.user_new_embeddings, self.item_new_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
            self.u_embeddings = tf.nn.embedding_lookup(self.user_new_embeddings, self.input_user)
            self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_new_embeddings, self.input_item_pos)
            self.neg_j_embeddings = tf.nn.embedding_lookup(self.item_new_embeddings, self.input_item_neg)
            
    def _create_loss(self):
        with tf.name_scope("loss"): 
            self.output = tf.reduce_sum(tf.multiply(self.u_embeddings, self.pos_i_embeddings), axis=1)
            output_neg = tf.reduce_sum(tf.multiply(self.u_embeddings, self.neg_j_embeddings), axis=1)
            regularizer = self.reg * l2_loss(self.u_embeddings, self.pos_i_embeddings, self.neg_j_embeddings)

            self.loss = Learner.pairwise_loss(self.loss_function,self.output - output_neg)+ regularizer
                
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = Learner.optimizer(self.learner, self.loss, self.learning_rate)
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
    
    def adjacient_matrix(self, self_connection=False):
        A = np.zeros([self.num_users+self.num_items, self.num_users+self.num_items], dtype=np.float32)
        A[:self.num_users, self.num_users:] = self.graph
        A[self.num_users:, :self.num_users] = self.graph.transpose()
        if self_connection == True:
            return np.identity(self.num_users+self.num_items,dtype=np.float32) + A
        return A

    def degree_matrix(self):
        degree = np.sum(self.A, axis=1, keepdims=False)
        #degree = np.diag(degree)
        return degree

    def laplacian_matrix(self, normalized=False):
        if normalized == False:
            return self.D - self.A

        temp = np.dot(np.diag(np.power(self.D, -1)), self.A)
        #temp = np.dot(temp, np.power(self.D, -0.5))
        return np.identity(self.num_users+self.num_items,dtype=np.float32) - temp
        
    def train_model(self):
        logger.info(self.evaluator.metrics_info())
        for epoch in  range(1,self.num_epochs+1):
            # Generate training instances
            user_input, item_input_pos, item_input_neg = DataGenerator._get_pairwise_all_data(self.dataset)
            data_iter = DataIterator(user_input, item_input_pos, item_input_neg,
                                         batch_size=self.batch_size, shuffle=True)
            
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for bat_users, bat_items_pos, bat_items_neg in data_iter:
                    feed_dict = {self.input_user: bat_users,
                                 self.input_item_pos: bat_items_pos,
                                 self.input_item_neg: bat_items_neg}
                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
    
            logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/num_training_instances,
                                                             time()-training_start_time))
            if epoch % self.verbose == 0:
                logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
                
    @timer
    def evaluate(self):
        self._cur_user_embeddings, self._cur_item_embeddings = self.sess.run([self.user_new_embeddings, self.item_new_embeddings])
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_userids):
        if candidate_items_userids is None:
            user_embed = self._cur_user_embeddings[user_ids]
            ratings = np.matmul(user_embed, self._cur_item_embeddings.T)
        else:
            ratings = []
            for userid, items_by_userid in zip(user_ids, candidate_items_userids):
                user_embed = self._cur_user_embeddings[userid]
                items_embed = self._cur_item_embeddings[items_by_userid]
                ratings.append(np.squeeze(np.matmul(user_embed, items_embed.T)))
            
        return ratings
