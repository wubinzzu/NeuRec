'''
Reference: Lei, Zheng, et al. "Spectral Collaborative Filtering." in RecSys2018
@author: wubin
'''
import tensorflow as tf
import numpy as np
from time import time
from neurec.util import data_gen,learner, tool
import configparser
from neurec.evaluation import Evaluate
from neurec.model.AbstractRecommender import AbstractRecommender
class SpectralCF(AbstractRecommender):
    def __init__(self,sess,dataset):
        config = configparser.ConfigParser()
        config.read("conf/SpectralCF.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("SpectralCF arguments: %s " %(self.conf))
        self.learning_rate = float(self.conf["learning_rate"])
        self.learner = self.conf["learner"]
        self.topK = int(self.conf["topk"])
        self.batch_size= int(self.conf["batch_size"])
        self.num_layers = int(self.conf["num_layers"])
        self.activation = self.conf["activation"]
        self.embedding_size = int(self.conf["embedding_size"])
        self.num_epochs=int(self.conf["epochs"])
        self.reg=float(self.conf["reg"])
        self.loss_function=self.conf["loss_function"]
        self.dropout=float(self.conf["dropout"])
        self.verbose=int(self.conf["verbose"])
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items  
        self.graph = dataset.trainMatrix.toarray()
        self.A = self.adjacient_matrix(self_connection=True)
        self.D = self.degree_matrix()
        self.L = self.laplacian_matrix(normalized=True)
        self.lamda, self.U = np.linalg.eig(self.L)
        self.lamda = np.diag(self.lamda)
        self.sess=sess        

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.input_user = tf.placeholder(dtype=tf.int32, shape=[None,])
            self.input_item = tf.placeholder(dtype=tf.int32, shape=[None,])
            self.input_item_neg = tf.placeholder(dtype=tf.int32, shape=[None,])
    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now   
            self.user_embeddings = tf.Variable(
            tf.random_normal([self.num_users, self.embedding_size], mean=0, stddev=0.01, dtype=tf.float32),
            name='user_embeddings')
            self.item_embeddings = tf.Variable(
            tf.random_normal([self.num_items, self.embedding_size], mean=0, stddev=0.01, dtype=tf.float32),
            name='item_embeddings')
        
        self.filters = []
        for _ in range(self.num_layers):
            self.filters.append(
                tf.Variable(tf.random_normal([self.embedding_size, self.embedding_size],
                    mean=0, stddev=0.01, dtype=tf.float32)))
    
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
                embeddings = tool.activation_function(self.activation, (tf.matmul(embeddings, self.filters[k])))
                all_embeddings += [embeddings]
            all_embeddings = tf.concat(all_embeddings, 1)
            self.user_new_embeddings, self.item_new_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
            self.u_embeddings = tf.nn.embedding_lookup(self.user_new_embeddings, self.input_user)
            self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_new_embeddings, self.input_item)
            self.neg_j_embeddings = tf.nn.embedding_lookup(self.item_new_embeddings, self.input_item_neg)
            
    def _create_loss(self):
        with tf.name_scope("loss"): 
            self.output = tf.reduce_sum(tf.multiply(self.u_embeddings, self.pos_i_embeddings), axis=1)
            output_neg = tf.reduce_sum(tf.multiply(self.u_embeddings, self.neg_j_embeddings), axis=1)
            regularizer = self.reg * ( tf.reduce_sum(tf.square(self.u_embeddings)) \
                + tf.reduce_sum(tf.square(self.pos_i_embeddings)) + tf.reduce_sum(tf.square(self.neg_j_embeddings)))
            
            self.loss = learner.pairwise_loss(self.loss_function,self.output - output_neg)+ regularizer
                
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
    
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
                feed_dict = {self.input_user:bat_users,self.input_item:bat_items_pos,\
                            self.input_item_neg:bat_items_neg}
                      
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss  
            
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)
                
    def predict(self, user_id, items):
        users = np.full(len(items), user_id, dtype=np.int32)
        return self.sess.run(self.output, feed_dict={self.input_user: users, self.input_item: items})