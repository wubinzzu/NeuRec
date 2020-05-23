"""
Reference: Tong Zhao et al., "Leveraging Social Connections to Improve 
Personalized Ranking for Collaborative Filtering." in CIKM 2014
@author: wubin
"""
import tensorflow as tf
import numpy as np
from time import time
from util import learner, randint_choice, tool
from model.AbstractRecommender import SocialAbstractRecommender
from util import timer
from util.data_iterator import DataIterator
from util.tool import csr_to_user_dict
from util import l2_loss


class SBPR(SocialAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(SBPR, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.embedding_size = conf["embedding_size"]
        self.learner = conf["learner"]
        self.loss_function = conf["loss_function"]
        self.num_epochs = conf["num_epochs"]
        self.reg_mf = conf["reg_mf"]
        self.batch_size = conf["batch_size"]

        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.verbose = conf["verbose"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.userids = self.dataset.userids
        self.train_dict = csr_to_user_dict(self.dataset.train_matrix)
        # self.social_matrix = self._get_social_matrix()
        self.userSocialItemsSetList = self._get_SocialItemsSet()
        self.sess = sess

    def _get_SocialItemsSet(self):
        # find items rated by trusted neighbors only
        userSocialItemsSetList = {}
        for u, _ in self.train_dict.items():
            trustors = self.social_matrix[u].indices
            items = [item for f_u in trustors for item in self.train_dict[f_u] if item not in self.train_dict[u]]
            items = set(items)
            if len(items) > 0:
                userSocialItemsSetList[u] = list(items)
        return userSocialItemsSetList

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None], name="item_input_pos")
            self.item_input_social = tf.placeholder(tf.int32, shape=[None], name="item_input_social")
            self.suk = tf.placeholder(tf.float32, shape=[None], name="suk")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            initializer = tool.get_initializer(self.init_method, self.stddev)
            self.user_embeddings = tf.Variable(initializer([self.num_users, self.embedding_size]),
                                               name='user_embeddings', dtype=tf.float32)  # (users, embedding_size)
            self.item_embeddings = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                               name='item_embeddings', dtype=tf.float32)  # (items, embedding_size)
            self.bias = tf.Variable(initializer([self.num_items]), name='bias')

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, item_input)
            # item_bias = tf.nn.embedding_lookup(self.bias, item_input)
            item_bias = tf.gather(self.bias, item_input)
            output = tf.reduce_sum(tf.multiply(user_embedding, item_embedding), 1) + item_bias
            return user_embedding, item_embedding, item_bias, output

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            p1, q1, b1, self.output = self._create_inference(self.item_input_pos)
            _, q2, b2, output_social = self._create_inference(self.item_input_social)
            _, q3, b3, output_neg = self._create_inference(self.item_input_neg)
            result1 = tf.divide(self.output - output_social, self.suk)
            result2 = output_social - output_neg
            self.loss = learner.pairwise_loss(self.loss_function, result1) + \
                        learner.pairwise_loss(self.loss_function, result2) + \
                        self.reg_mf * l2_loss(p1, q2, q1, q3, b1, b2, b3)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()

    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.num_epochs):
            user_input, item_input_pos, item_input_social, item_input_neg, suk_input = self._get_pairwise_all_data()
            data_iter = DataIterator(user_input, item_input_pos, item_input_social, item_input_neg, suk_input,
                                     batch_size=self.batch_size, shuffle=True)
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for bat_users, bat_items_pos, bat_items_social, bat_items_neg, bat_suk_input in data_iter:
                feed_dict = {self.user_input: bat_users, self.item_input_pos: bat_items_pos,
                             self.item_input_social: bat_items_social,
                             self.item_input_neg: bat_items_neg, self.suk: bat_suk_input}
                      
                loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                total_loss += loss
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/num_training_instances,
                                                             time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    def _get_pairwise_all_data(self):
        user_input, item_input_pos, item_input_social, item_input_neg, suk_input = [], [], [], [], []
        num_items = self.dataset.num_items

        for u, pos_item in self.train_dict.items():
            if u not in self.userSocialItemsSetList:
                continue
            # pos_item = pos_item.indices.tolist()
            pos_len = len(pos_item)
            user_input.extend([u]*pos_len)
            item_input_pos.extend(pos_item)

            socialItemsList = self.userSocialItemsSetList[u]
            # a, size = None, replace = True, p = None, exclusion = None
            neg_excl = np.concatenate([socialItemsList, list(pos_item)], axis=0)

            neg_item = randint_choice(num_items, pos_len, replace=True, exclusion=neg_excl)
            item_input_neg.extend(neg_item)
            social_item = np.random.choice(socialItemsList, size=pos_len)
            item_input_social.extend(social_item)

            trustedUserIdices = self.social_matrix[u].indices
            socialWeight_bool = [[1 if k in self.train_dict[f_u] else 0 for f_u in trustedUserIdices] for k in social_item]
            socialWeight = np.sum(socialWeight_bool, axis=-1) + 1
            suk_input.extend(socialWeight)
            
        return user_input, item_input_pos, item_input_social, item_input_neg, suk_input
            
    @timer
    def evaluate(self):
        self._cur_user_embeddings, self._cur_item_embeddings = self.sess.run([self.user_embeddings, self.item_embeddings])
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
