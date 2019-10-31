"""
Created on Aug 8, 2016
Processing datasets. 
@author: Xiangnan He (xiangnanhe@gmail.com)
"""
import numpy as np
from neurec.data.DataSplitter import DataSplitter
from neurec.data.GivenData import GivenData
from neurec.util.tool import randint_choice
import scipy.sparse as sp
from neurec.util.singleton import Singleton
from importlib import util
import os

class Dataset(metaclass=Singleton):
    def __init__(self, dataset_path, dataset_name, data_format, splitter, separator, threshold, evaluate_neg, splitter_ratio=[0.8,0.2]):
        """
        Constructor
        """
        if (dataset_path == 'neurec'):
            neurec_path = util.find_spec('neurec', package='neurec').submodule_search_locations[0]
            dataset_path = os.path.join(neurec_path, 'dataset', dataset_name)
        
        self.path = dataset_path
        self.dataset_name = dataset_name
        self.data_format = data_format
        self.separator= separator
        self.threshold = threshold
        self.splitter_ratio = splitter_ratio
        self.evaluate_neg = evaluate_neg
        self.splitter = splitter

        if splitter == "given":
            data_splitter = GivenData(self.path, self.data_format, self.separator, self.threshold)
            
        elif splitter == "loo" or splitter == "ratio":
            data_splitter = DataSplitter(self.path, self.splitter, self.data_format, self.separator, self.threshold, self.splitter_ratio)
            
        else:
            raise ValueError("please choose a correct splitter")
        
        self.train_matrix, self.test_matrix, self.time_matrix, self.userids, self.itemids = data_splitter.load_data()
   
        self.num_users, self.num_items = self.train_matrix.shape
        self.negative_matrix = self.get_negatives(self.evaluate_neg)
             
    def get_negatives(self, evaluate_neg):
        if evaluate_neg > 0:
            user_list = []
            neg_item_list = []
            for u in np.arange(self.num_users):
                items_by_u = self.train_matrix[u].indices.tolist() + self.test_matrix[u].indices.tolist()
                neg_items = randint_choice(self.num_items, evaluate_neg, replace=False, exclusion=items_by_u).tolist()
                neg_item_list.extend(neg_items)
                user_list.extend(len(neg_items)*[u])
            negatives = sp.csr_matrix(([1] * len(user_list), (user_list, neg_item_list)),
                                      shape=(self.num_users, self.num_items))
        else:
            negatives = None
        return negatives
