from model.AbstractRecommender import SocialAbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner, tool
from util.tool import csr_to_user_dict
from util import timer
from util import l2_loss
from data import PointwiseSampler


class DiffNet(SocialAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(DiffNet, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.embedding_size = conf["embedding_size"]
        self.learner = conf["learner"]
        self.loss_function = conf["loss_function"]
        self.num_epochs = conf["epochs"]
        self.reg_mf = conf["reg_mf"]
        self.batch_size = conf["batch_size"]
        self.num_negatives = conf["num_negatives"]
        self.user_feature_file = conf["user_feature_file"]
        self.item_feature_file = conf["item_feature_file"]
        self.feature_dimension = conf["feature_dimension"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.verbose = conf["verbose"]
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.userids = dataset.userids
        self.itemids = dataset.itemids
        self.dataset = dataset
        self.trainMatrix = dataset.trainMatrix
        self.trainDict = csr_to_user_dict(self.trainMatrix)
        self.social_matrix = self.social_matrix + self.social_matrix.transpose()

        self.consumed_items_sparse_matrix, self.social_neighbors_sparse_matrix = self.input_supply()
        self.sess = sess
        
    def input_supply(self):
        consumed_items_indices_list = []
        consumed_items_values_list = []
        for (u, i) in self.trainMatrix.keys():
            consumed_items_indices_list.append([u, i])
            consumed_items_values_list.append(1.0/len(self.trainDict[u]))
        self.consumed_items_indices_list = np.array(consumed_items_indices_list).astype(np.int32)
        self.consumed_items_values_list = np.array(consumed_items_values_list).astype(np.float32)

        social_neighbors_indices_list = []
        social_neighbors_values_list = []
        for u in range(self.num_users):
            friends = self.social_matrix[u].indices
            for v in friends:
                social_neighbors_indices_list.append([u,v])
                social_neighbors_values_list.append(1.0/len(friends))
                    
        self.social_neighbors_indices_list = np.array(social_neighbors_indices_list).astype(np.int32)
        self.social_neighbors_values_list = np.array(social_neighbors_values_list).astype(np.float32)

        # prepare sparse matrix, in order to compute user's embedding from social neighbors and consumed items
        self.social_neighbors_dense_shape = np.array([self.num_users, self.num_users]).astype(np.int32)
        self.consumed_items_dense_shape = np.array([self.num_users, self.num_items]).astype(np.int32)

        self.social_neighbors_sparse_matrix = tf.SparseTensor(indices=self.social_neighbors_indices_list,
                                                              values=self.social_neighbors_values_list,
                                                              dense_shape=self.social_neighbors_dense_shape)
        self.consumed_items_sparse_matrix = tf.SparseTensor(indices=self.consumed_items_indices_list,
                                                            values=self.consumed_items_values_list,
                                                            dense_shape=self.consumed_items_dense_shape)
        return self.consumed_items_sparse_matrix, self.social_neighbors_sparse_matrix
    
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None], name="item_input_pos")
            self.labels = tf.placeholder(tf.float32, shape=[None], name="labels_input")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            initializer = tool.get_initializer(self.init_method, self.stddev)
               
            self.user_embedding = tf.Variable(initializer([self.num_users, self.embedding_size]),
                                              name='user_embedding')
            self.item_embedding = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                              name='item_embedding')
            
        user_review_vectors = np.zeros((self.num_users, self.feature_dimension))
        with open(self.user_feature_file, 'r') as f:
            for line in f.readlines():
                user_idx, data = line.strip().split("::::")
                if user_idx in self.userids:
                    inner_user_idx = self.userids[user_idx]
                    user_review_vectors[inner_user_idx] = eval(data)
            
        self.user_review_vector_matrix = tf.constant(user_review_vectors, dtype=tf.float32)

        item_review_vectors = np.zeros((self.num_items, self.feature_dimension))
        with open(self.item_feature_file, 'r') as f:
            for line in f.readlines():
                item_idx, data = line.strip().split("::::")
                if item_idx in self.itemids:
                    inner_item_idx = self.itemids[item_idx]
                    item_review_vectors[inner_item_idx] = eval(data)
            
        self.item_review_vector_matrix = tf.constant(item_review_vectors, dtype=tf.float32)
        self.reduce_dimension_layer = tf.layers.Dense(self.embedding_size, activation=tf.nn.sigmoid,
                                                      name='reduce_dimension_layer')
        
        self.item_fusion_layer = tf.layers.Dense(self.embedding_size, activation=tf.nn.sigmoid,
                                                 name='item_fusion_layer')
        self.user_fusion_layer = tf.layers.Dense(self.embedding_size, activation=tf.nn.sigmoid,
                                                 name='user_fusion_layer')

    def convertDistribution(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1])
        y = (x - mean) * 0.1 / tf.sqrt(var)
        return y
      
    def generateUserEmbeddingFromSocialNeighbors(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(self.social_neighbors_sparse_matrix,
                                                                             current_user_embedding)
        return user_embedding_from_social_neighbors

    def generateUserEmebddingFromConsumedItems(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(self.consumed_items_sparse_matrix,
                                                                           current_item_embedding)
        return user_embedding_from_consumed_items
    
    def _create_inference(self):
        with tf.name_scope("inference"):
            first_user_review_vector_matrix = self.convertDistribution(self.user_review_vector_matrix)
            first_item_review_vector_matrix = self.convertDistribution(self.item_review_vector_matrix)
            
            self.user_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_user_review_vector_matrix)
            self.item_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_item_review_vector_matrix)
    
            second_user_review_vector_matrix = self.convertDistribution(self.user_reduce_dim_vector_matrix)
            second_item_review_vector_matrix = self.convertDistribution(self.item_reduce_dim_vector_matrix)
            
            # compute item embedding
            #self.fusion_item_embedding = self.item_fusion_layer(\
            #   tf.concat([self.item_embedding, second_item_review_vector_matrix], 1))
            self.final_item_embedding = self.fusion_item_embedding = \
                self.item_embedding + second_item_review_vector_matrix  # TODO ?
            #self.final_item_embedding = self.fusion_item_embedding = second_item_review_vector_matrix
    
            # compute user embedding
            user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems(self.final_item_embedding)
    
            #self.fusion_user_embedding = self.user_fusion_layer(\
            #    tf.concat([self.user_embedding, second_user_review_vector_matrix], 1))
            self.fusion_user_embedding = self.user_embedding  # + second_user_review_vector_matrix
            first_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(self.fusion_user_embedding)
            second_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(first_gcn_user_embedding)
    
            self.final_user_embedding = second_gcn_user_embedding + user_embedding_from_consumed_items
            
            # embedding look up
            self.latest_user_latent = tf.nn.embedding_lookup(self.final_user_embedding, self.user_input)
            self.latest_item_latent = tf.nn.embedding_lookup(self.final_item_embedding, self.item_input)
            
#             self.output = tf.sigmoid(tf.reduce_sum(tf.multiply(self.latest_user_latent, self.latest_item_latent),1))
            self.output = tf.reduce_sum(tf.multiply(self.latest_user_latent, self.latest_item_latent), 1)
            
    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            # reg = l2_regularizer(self.reg_mf)
            #
            # reg_var = apply_regularization(reg, self.user_embedding + self.item_embedding)
            
            self.loss = tf.losses.sigmoid_cross_entropy(self.labels, self.output) + \
            self.reg_mf*l2_loss(self.latest_user_latent, self.latest_item_latent)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
                
    def build_graph(self):
        self._create_placeholders()  
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()

    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        data_iter = PointwiseSampler(self.dataset, neg_num=self.num_negatives, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.num_epochs+1):
            total_loss = 0.0
            start_time = time()
            num_instances = len(data_iter)
            for bat_users, bat_items, bat_labels in data_iter:
                feed_dict = {self.user_input: bat_users,
                             self.item_input: bat_items,
                             self.labels: bat_labels}
                loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                total_loss += loss
            self.logger.info("[iter %d : loss : %f, time: %f]" %
                             (epoch, total_loss / num_instances,time() - start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        self._cur_user_embeddings, self._cur_item_embeddings = \
            self.sess.run([self.latest_user_latent, self.latest_item_latent])
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_userids):
        if candidate_items_userids is None:
            user_embed = self._cur_user_embeddings[user_ids]
            ratings = np.matmul(user_embed, self._cur_item_embeddings.T)
        else:
            ratings = []
            for user_id, items_by_user_id in zip(user_ids, candidate_items_userids):
                user_embed = self._cur_user_embeddings[user_id]
                items_embed = self._cur_item_embeddings[items_by_user_id]
                ratings.append(np.squeeze(np.matmul(user_embed, items_embed.T)))
            
        return ratings
