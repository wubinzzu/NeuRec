'''
Reference: Yifan Hu et al., "Collaborative Filtering for Implicit Feedback Datasets" in ICDM 2008.
@author: wubin
'''
from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
from time import time
from neurec.evaluation import Evaluate
from neurec.model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
from neurec.util.properties import Properties

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class WRMF(AbstractRecommender):
    properties = [
        "embedding_size",
        "alpha",
        "topk",
        "epochs",
        "reg_mf",
        "verbose"
    ]

    def __init__(self, sess, dataset):
        super().__init__(**kwds)

        self.embedding_size = self.conf["embedding_size"]
        self.alpha = self.conf["alpha"]
        self.topK = self.conf["topk"]
        self.num_epochs= self.conf["epochs"]
        self.reg_mf = self.conf["reg_mf"]
        self.verbose= self.conf["verbose"]

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

        self.Cui = np.zeros(shape=[self.num_users, self.num_items], dtype=np.float32)
        self.Pui = np.zeros(shape=[self.num_users, self.num_items], dtype=np.float32)
        for (u, i) in self.dataset.trainMatrix.keys():
            self.Cui[u,i] = self.alpha
            self.Pui[u,i] = 1.0
        self.lambda_eye = self.reg_mf * tf.eye(self.embedding_size)

    def _create_placeholders(self):
        self.user_id = tf.compat.v1.placeholder(tf.int32, [1])
        self.Cu = tf.compat.v1.placeholder(tf.float32, [self.num_items, 1])
        self.Pu = tf.compat.v1.placeholder(tf.float32, [self.num_items, 1])

        self.item_id = tf.compat.v1.placeholder(tf.int32, [1])
        self.Ci = tf.compat.v1.placeholder(tf.float32, [self.num_users, 1])
        self.Pi = tf.compat.v1.placeholder(tf.float32, [self.num_users, 1])

    def _create_variables(self):
        self.user_embeddings = tf.Variable(tf.random.normal([self.num_users, self.embedding_size], stddev=0.01))
        self.item_embeddings = tf.Variable(tf.random.normal([self.num_items, self.embedding_size], stddev=0.01))

    def _create_optimizer(self):
        YTY = tf.matmul(self.item_embeddings, self.item_embeddings, transpose_a=True)
        YTCuIY = tf.matmul(self.item_embeddings, tf.multiply(self.Cu, self.item_embeddings), transpose_a=True)
        YTCupu = tf.matmul(self.item_embeddings, tf.multiply(self.Cu+1, self.Pu), transpose_a=True)
        xu = tf.linalg.solve(YTY + YTCuIY + self.lambda_eye, YTCupu)
        self.update_user = tf.scatter_update(self.user_embeddings, self.user_id, tf.transpose(xu))

        XTX = tf.matmul(self.user_embeddings, self.user_embeddings, transpose_a=True)
        XTCIIX = tf.matmul(self.user_embeddings, tf.multiply(self.Ci, self.user_embeddings), transpose_a=True)
        XTCIpi = tf.matmul(self.user_embeddings, tf.multiply(self.Ci+1, self.Pi), transpose_a=True)
        xi = tf.linalg.solve(XTX + XTCIIX + self.lambda_eye, XTCIpi)
        self.update_item = tf.scatter_update(self.item_embeddings, self.item_id, tf.transpose(xi))

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_optimizer()

        #---------- training process -------
    def train_model(self):
        for epoch in  range(self.num_epochs):
            training_start_time = time()
            print('solving for user vectors...')
            for userid in range(self.num_users):
                feed = {self.user_id: [userid],
                        self.Pu: self.Pui[userid].T.reshape([-1,1]),
                        self.Cu: self.Cui[userid].T.reshape([-1,1])}
                self.sess.run(self.update_user, feed_dict=feed)

            print('solving for item vectors...')
            for itemid in range(self.num_items):
                feed = {self.item_id: [itemid],
                        self.Pi: self.Pui[:,itemid].reshape([-1,1]),
                        self.Ci: self.Cui[:,itemid].reshape([-1,1])}
                self.sess.run(self.update_item, feed_dict=feed)

            print ('iteration %i finished in %f seconds' % (epoch + 1, time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)

    def predict(self, user_id, items):
        user_embeddings, item_embeddings = self.sess.run([self.user_embeddings, self.item_embeddings])
        user_embedding = user_embeddings[user_id]
        item_embedding = item_embeddings[items]
        predictions = user_embedding.dot(item_embedding.T)
        return predictions
