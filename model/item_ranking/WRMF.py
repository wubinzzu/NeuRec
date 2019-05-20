'''
Reference: Yifan Hu et al., "Collaborative Filtering for Implicit Feedback Datasets" in ICDM 2008.
@author: wubin
'''
from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
from time import time
import configparser
import scipy.sparse as sparse
from numpy.linalg import solve
from evaluation import Evaluate
from model.AbstractRecommender import AbstractRecommender
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class WRMF(AbstractRecommender):
    def __init__(self,dataset):  
        config = configparser.ConfigParser()
        config.read("conf/WRMF.properties")
        self.conf=dict(config.items("hyperparameters"))
        print("WRMF arguments: %s " %(self.conf))
        self.embedding_size = int(self.conf["embedding_size"])
        self.alpha = float(self.conf["alpha"])
        self.topK = int(self.conf["topk"])
        self.num_epochs= int(self.conf["epochs"])
        self.reg_mf = float(self.conf["reg_mf"])
        self.verbose= int(self.conf["verbose"])
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items 
    def _create_variables(self):
        self.user_embeddings = np.mat(np.random.normal(size=(self.num_users,self.embedding_size),scale=0.01))
        self.item_embeddings = np.mat(np.random.normal(size=(self.num_items,self.embedding_size),scale=0.01))
    
    def build_graph(self):
        self._create_variables()
        self.Cui = sparse.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        self.Pui = sparse.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        for (u, i) in self.dataset.trainMatrix.keys():
            self.Cui[u,i] = self.alpha
            self.Pui[u,i] = 1.0   
             
        #---------- training process -------
    def train_model(self):
        for epoch in  range(self.num_epochs):
            training_start_time = time()
            print ('solving for user vectors...')
            YTY = np.transpose(self.item_embeddings).dot(self.item_embeddings)
            eye = np.eye(self.num_items)
            lambda_eye = self.reg_mf * np.eye(self.embedding_size)
            for userid in range(self.num_users):
                pu= self.Pui[userid].toarray()
                self.Cui[userid].toarray()
                Cu = np.diag(np.reshape(self.Cui[userid].toarray(),self.num_items))
                YTCuIY = self.item_embeddings.T.dot(Cu).dot(self.item_embeddings)
                YTCupu = self.item_embeddings.T.dot(Cu + eye).dot(pu.T)
                   
                xu = solve(YTY + YTCuIY + lambda_eye,YTCupu)
                self.user_embeddings[userid] = xu.T   
            print ('solving for item vectors...')
            XTX = np.transpose(self.user_embeddings).dot(self.user_embeddings)
            eye = np.eye(self.num_users)
            for itemid in range(self.num_items):
                pi= self.Pui[:,itemid].toarray()
                Ci = np.diag(np.reshape(self.Cui[:,itemid].T.toarray(),self.num_users))
                XTCIIX = self.user_embeddings.T.dot(Ci).dot(self.user_embeddings)
                XTCIpi = self.user_embeddings.T.dot(Ci + eye).dot(pi)
                   
                xi = solve(XTX + XTCIIX + lambda_eye,XTCIpi)
                self.item_embeddings[itemid] = xi.T   
           
            print ('iteration %i finished in %f seconds' % (epoch + 1, time()-training_start_time))
            if epoch %self.verbose == 0:
                Evaluate.test_model(self,self.dataset)
                
    def predict(self, user_id, items):
        predictions = []
        user_embedding = self.user_embeddings[user_id]
        for item_id in items:
            item_embedding = self.item_embeddings[item_id]
            predictions.append(user_embedding.dot(item_embedding.T))
        return predictions
                