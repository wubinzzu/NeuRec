"""
Created on Aug 8, 2016
Processing datasets. 
@author: Xiangnan He (xiangnanhe@gmail.com)
"""
import numpy as np
from data.DataSplitter import DataSplitter
from data.GivenData import GivenData
from util.Tool import randint_choice
import scipy.sparse as sp


class Dataset(object):
    def __init__(self, conf):
        """
        Constructor
        """
        path = conf["data.input.path"]
        dataset_name = conf["data.input.dataset"]
        data_format = conf["data.column.format"]
        splitter = conf["data.splitter"]
        self.separator = conf["data.convert.separator"]
        threshold = conf["data.convert.binarize.threshold"]
        evaluate_neg = conf["rec.evaluate.neg"]
        splitter_ratio = conf["data.splitterratio"]

        path = path + dataset_name
        self.dataset_name = dataset_name
        self.splitter = splitter
        
        if splitter == "given":
            data_splitter = GivenData(path, data_format, self.separator, threshold)
            
        elif splitter == "loo" or splitter == "ratio":
            data_splitter = DataSplitter(path, splitter, data_format, self.separator, threshold, splitter_ratio)
            
        else:
            raise ValueError("please choose a correct splitter")
        
        self.train_matrix, self.test_matrix, self.time_matrix, self.userids, self.itemids = data_splitter.load_data()
   
        self.num_users, self.num_items = self.train_matrix.shape
        self.negative_matrix = self.get_negatives(evaluate_neg)
             
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
