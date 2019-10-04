'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
from neurec.data.LeaveOneOutDataSplitter import LeaveOneOutDataSplitter
from neurec.data.HoldOutDataSplitter import HoldOutDataSplitter
from neurec.data.GivenData import GivenData
from neurec.util.singleton import Singleton

class Dataset(metaclass=Singleton):

    def __init__(self, dataset_path, dataset_name, data_format, splitter,separator,threshold,evaluate_neg,splitterRatio=[0.8,0.2]):
        '''
        Constructor
        '''
        self.path = dataset_path
        self.dataset_name = dataset_name
        self.data_format = data_format
        self.separator= separator
        self.threshold = threshold
        self.splitterRatio=splitterRatio
        self.evaluate_neg = evaluate_neg
        self.splitter=splitter
        self.num_users = 0
        self.num_items = 0
        self.trainMatrix = None
        self.trainDict =  None
        self.testMatrix =  None
        self.testNegatives =  None
        self.timeMatrix = None
        self.userseq = None
        self.userids = None
        self.itemids = None
        if splitter == "loo" :
            loo = LeaveOneOutDataSplitter(self.path, self.dataset_name, self.data_format,self.separator, self.threshold)
            self.trainMatrix,self.trainDict,self.testMatrix,\
            self.userseq,self.userids,self.itemids,self.timeMatrix = loo.load_data_by_user_time()
            self.num_users = self.trainMatrix.shape[0]
            self.num_items = self.trainMatrix.shape[1]
        elif splitter == "ratio" :
            hold_out = HoldOutDataSplitter(self.path, self.dataset_name,self.data_format, self.separator,self.threshold,self.splitterRatio)
            self.trainMatrix,self.trainDict,self.testMatrix,\
            self.userseq,self.userids,self.itemids,self.timeMatrix =\
            hold_out.load_data_by_user_time()
            self.num_users = self.trainMatrix.shape[0]
            self.num_items = self.trainMatrix.shape[1]
        elif splitter == "given":
            given = GivenData(self.path, self.dataset_name,self.separator,self.threshold)
            self.trainMatrix,self.trainDict,self.testMatrix,\
            self.userseq,self.userids,self.itemids,self.timeMatrix =\
            given.load_pre_splitter_data()
            self.num_users = self.trainMatrix.shape[0]
            self.num_items = self.trainMatrix.shape[1]
        else :
            print("please choose a splitter")

        self.testNegatives = self.get_negatives()
        self.num_users = self.trainMatrix.shape[0]
        self.num_items = self.trainMatrix.shape[1]

    def get_negatives(self):
        negatives = {}
        for u in np.arange(self.num_users):
            negative_per_user =[]
            if(self.evaluate_neg>0):
                for _ in np.arange(self.evaluate_neg):
                    neg_item_id = np.random.randint(0,self.num_items)
                    while (u,neg_item_id) in self.trainMatrix.keys() or  (u,neg_item_id) in self.testMatrix.keys() \
                          or neg_item_id in negative_per_user:
                        neg_item_id = np.random.randint(0, self.num_items)
                    negative_per_user.append(neg_item_id)
                negatives[u] = negative_per_user
                negative_per_user =[]
            else :
                negatives=None
        return  negatives
