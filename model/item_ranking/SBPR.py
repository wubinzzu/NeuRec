'''
Reference: Tong Zhao et al., "Leveraging Social Connections to Improve 
Personalized Ranking for Collaborative Filtering." in CIKM 2014
@author: wubin
'''
from __future__ import absolute_import
from __future__ import division
import scipy.sparse as sp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from time import time
from util import learner
from evaluation import Evaluate
from model.AbstractRecommender import AbstractRecommender
import configparser
from util.Logger import logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
def random_choice(a, size=None, replace=True, p=None, exclusion=None):
    # TODO exclusion is element, not the index
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = np.ndarray.flatten(p)
        p[exclusion] = 0
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample

class SBPR(AbstractRecommender):
    def __init__(self,sess,dataset):
        config = configparser.ConfigParser()
        config.read("conf/SBPR.properties")
        self.conf = dict(config.items("hyperparameters"))
        self.socialpath = self.conf["socialpath"]
        self.learning_rate = eval(self.conf["learning_rate"])
        self.embedding_size = eval(self.conf["embedding_size"])
        self.learner = self.conf["learner"]
        self.loss_function = self.conf["loss_function"]
        self.topK = eval(self.conf["topk"])
        self.num_epochs= eval(self.conf["num_epochs"])
        self.reg_mf = eval(self.conf["reg_mf"])
        self.batch_size = eval(self.conf["batch_size"])
        self.verbose = eval(self.conf["verbose"])
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.userids = self.dataset.userids
        self.dataset_name = dataset.dataset_name
        self.userouterids = self.userids.keys()
        trainMatrix = self.dataset.trainMatrix.tocsr()
        self.train_dict = {u: set(pos_item.indices) for u, pos_item in enumerate(trainMatrix)}
        self.socialMatrix=self._get_social_data()
        self.userSocialItemsSetList = self._get_SocialItemsSet_sun()
        logger.info("init finished")
        self.sess = sess

    def _get_social_data(self):
        social_users = np.genfromtxt(self.socialpath, dtype=None, names=["user0", "user1"], delimiter=',')
        users_key = np.array(list(self.userids.keys()))
        user0 = social_users["user0"].astype(np.str)
        index = np.in1d(user0, users_key)
        social_users = social_users[index]
    
        user1 = social_users["user1"].astype(np.str)
        index = np.in1d(user1, users_key)
        social_users = social_users[index]
    
        user0 = social_users["user0"].astype(np.str)
        user0_id = [self.userids[u] for u in user0]
        user1 = social_users["user1"].astype(np.str)
        user1_id = [self.userids[u] for u in user1]
        social_matrix = sp.csr_matrix(([1]*len(user0_id), (user0_id, user1_id)),
                                      shape=(self.num_users, self.num_users)) 
        return social_matrix

    def _get_SocialItemsSet_sun(self):
        #find items rated by trusted neighbors only
        userSocialItemsSetList = {}
        for u, _ in self.train_dict.items():
            trustors = self.socialMatrix[u].indices
            items = [item for f_u in trustors for item in self.train_dict[f_u] if item not in self.train_dict[u]]
            items = set(items)
            if len(items) > 0:
                userSocialItemsSetList[u] = list(items)
        return userSocialItemsSetList

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None,], name = "user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape = [None,], name = "item_input_pos")
            self.item_input_social = tf.placeholder(tf.int32, shape = [None,], name = "item_input_social")
            self.suk = tf.placeholder(tf.float32, shape = [None,], name = "suk")
            self.item_input_neg = tf.placeholder(tf.int32, shape = [None,], name = "item_input_neg")
    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.user_embeddings = tf.Variable(tf.random_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='user_embeddings', dtype=tf.float32)  #(users, embedding_size)
            self.item_embeddings = tf.Variable(tf.random_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='item_embeddings', dtype=tf.float32)  #(items, embedding_size)
            self.bias = tf.Variable(tf.zeros(self.num_items),name='bias')
    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, item_input)
            item_bias = tf.nn.embedding_lookup(self.bias, item_input)
            output = tf.reduce_sum(tf.multiply(user_embedding, item_embedding),1) + item_bias
            return user_embedding, item_embedding,item_bias, output

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            p1, q1,b1,self.output= self._create_inference(self.item_input_pos)
            _, q2,b2, output_social = self._create_inference(self.item_input_social)
            _, q3,b3, output_neg = self._create_inference(self.item_input_neg)
            result1 = tf.divide(self.output - output_social,self.suk)
            result2 = output_social - output_neg
            self.loss = learner.pairwise_loss(self.loss_function,result1)+learner.pairwise_loss(self.loss_function,result2)+ self.reg_mf * ( tf.reduce_sum(tf.square(p1)) \
            + tf.reduce_sum(tf.square(q2)) + tf.reduce_sum(tf.square(q1))+tf.reduce_sum(tf.square(q3))+tf.reduce_sum(tf.square(b1))+tf.reduce_sum(tf.square(b2))+tf.reduce_sum(tf.square(b3)))
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
#---------- training process -------
    def train_model(self):
        for epoch in range(self.num_epochs):
            # Generate training instances
#             logger.info("get training data")
            user_input, item_input_pos,item_input_social,item_input_neg,suk_input = self._get_pairwise_all_data_sun()
#             logger.info("begin training")
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_input)
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
#                 print(num_batch)
                num_training_instances =len(user_input) 
                id_start = num_batch * self.batch_size
                id_end = (num_batch + 1) *self.batch_size
                if id_end>num_training_instances:
                    id_end=num_training_instances
                bat_users = user_input[id_start:id_end]
                bat_items_pos = item_input_pos[id_start:id_end]
                bat_items_social = item_input_social[id_start:id_end]
                bat_items_neg = item_input_neg[id_start:id_end]
                bat_suk_input = suk_input[id_start:id_end]
                feed_dict = {self.user_input:bat_users,self.item_input_pos:bat_items_pos,\
                            self.item_input_social:bat_items_social,\
                            self.item_input_neg:bat_items_neg,self.suk:bat_suk_input}
                      
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
            print("[iter %d : loss : %f, time: %f]" %(epoch+1,total_loss/num_training_instances,time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)

    def _get_pairwise_all_data(self):
        user_input, item_input_pos,item_input_social,item_input_neg,suk_input = [],[],[],[],[]
        trainMatrix = self.dataset.trainMatrix
        num_items = self.dataset.num_items
        for (u, i) in trainMatrix.keys():
            if u in self.userSocialItemsSetList:
                user_input.append(u)
                item_input_pos.append(i)
                socialItemsList = self.userSocialItemsSetList[u]
                j = np.random.randint(num_items)
                while (u, j) in trainMatrix.keys() or j in socialItemsList:
                    j = np.random.randint(num_items)
                item_input_neg.append(j)
                k = np.random.choice(socialItemsList)
                item_input_social.append(k)
                trustorIndices = self.socialMatrix[u].indices
                socialWeight = 0
                for trustedUserIdx in trustorIndices:
                    indices = trainMatrix[trustedUserIdx].tocsr().indices
                    if k in indices:
                        socialWeight += 1
                suk_input.append(socialWeight+1) 
        user_input = np.array(user_input, dtype=np.int32)
        item_input_pos = np.array(item_input_pos, dtype=np.int32)
        item_input_social = np.array(item_input_social, dtype=np.int32)
        item_input_neg = np.array(item_input_neg, dtype=np.int32)
        suk_input = np.array(suk_input, dtype=np.float32)
        num_training_instances = len(user_input)
        shuffle_index = np.arange(num_training_instances,dtype=np.int32)
        np.random.shuffle(shuffle_index)
        user_input=user_input[shuffle_index]
        item_input_pos=item_input_pos[shuffle_index]
        item_input_social=item_input_social[shuffle_index]
        item_input_neg=item_input_neg[shuffle_index]
        suk_input = suk_input[shuffle_index]
        return user_input, item_input_pos,item_input_social,item_input_neg,suk_input

    def _get_pairwise_all_data_sun(self):
        user_input, item_input_pos, item_input_social, item_input_neg, suk_input = [],[],[],[],[]

        num_items = self.dataset.num_items
        all_items = np.arange(num_items)

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

            neg_item = random_choice(all_items, pos_len, replace=True, exclusion=neg_excl)
            item_input_neg.extend(neg_item)
            social_item = np.random.choice(socialItemsList, size=pos_len)
            item_input_social.extend(social_item)

            trustedUserIdices = self.socialMatrix[u].indices
            socialWeight_bool = [[1 if k in self.train_dict[f_u] else 0 for f_u in trustedUserIdices]
                                 for k in social_item]
            socialWeight = np.sum(socialWeight_bool, axis=-1) + 1
            suk_input.extend(socialWeight)

        user_input = np.array(user_input, dtype=np.int32)
        item_input_pos = np.array(item_input_pos, dtype=np.int32)
        item_input_social = np.array(item_input_social, dtype=np.int32)
        item_input_neg = np.array(item_input_neg, dtype=np.int32)
        suk_input = np.array(suk_input, dtype=np.float32)
        num_training_instances = len(user_input)
        shuffle_index = np.arange(num_training_instances,dtype=np.int32)
        np.random.shuffle(shuffle_index)
        user_input=user_input[shuffle_index]
        item_input_pos=item_input_pos[shuffle_index]
        item_input_social=item_input_social[shuffle_index]
        item_input_neg=item_input_neg[shuffle_index]
        suk_input = suk_input[shuffle_index]
        return user_input, item_input_pos,item_input_social,item_input_neg,suk_input
            
    def predict(self, user_id, eval_items):
        users = np.full(len(eval_items), user_id, dtype=np.int32)
        return self.sess.run(self.output, feed_dict={self.user_input: users, self.item_input_pos: eval_items})  