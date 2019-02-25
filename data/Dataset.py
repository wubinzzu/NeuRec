'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import os
from data.LeaveOneOutDataSplitter import LeaveOneOutDataSplitter
from data.HoldOutDataSplitter import HoldOutDataSplitter
class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self,path,splitter,separator,evaluate_neg,dataset_name,isgivenTest=False,splitterRatio=[0.7,0.1,0.2]):
        '''
        Constructor
        '''
        self.path = path
        self.dataset_name = dataset_name
        self.separator= separator
        self.splitterRatio=splitterRatio
        self.evaluate_neg = evaluate_neg
        self.splitter=splitter
        self.num_users = 0
        self.num_items = 0
        self.trainMatrix = None
        self.trainDict =  None
        self.validDict =  None
        self.validMatrix =  None
        self.testMatrix =  None
        self.testNegatives =  None
        self.timeMatrix = None 
        self.userseq = None
        self.userids = None
        self.itemids = None
        if splitter == "loo" :
            loo = LeaveOneOutDataSplitter(self.path,self.separator)
            if isgivenTest.lower() == "true":
                self.trainMatrix = loo.load_training_file_as_matrix()
                self.trainDict = loo.load_training_file_as_list()
                self.validMatrix = loo.load_validrating_file_as_Matrix()
                self.testMatrix = loo.load_testrating_file_as_Matrix()
                self.num_users = self.trainMatrix.shape[0]
                self.num_items = self.trainMatrix.shape[1]
                if os.path.exists(self.path+".negative"):
                    self.testNegatives = loo.load_negative_file()
                else :
                    self.testNegatives = self.get_negatives()
            else :
                self.trainMatrix,self.trainDict,self.validDict,self.validMatrix,self.testMatrix,\
                self.userseq,self.userids,self.itemids,self.timeMatrix = loo.load_data_by_user_time()
                self.num_users = self.trainMatrix.shape[0]
                self.num_items = self.trainMatrix.shape[1]
                self.testNegatives = self.get_negatives()
        elif splitter == "ratio" :
            hold_out = HoldOutDataSplitter(self.path,self.separator,self.splitterRatio)
            if isgivenTest.lower() == "true":
                self.trainMatrix = hold_out.load_training_file_as_matrix()
                self.validMatrix = hold_out.load_validrating_file_as_matrix()
                self.testMatrix = hold_out.load_testrating_file_as_matrix()
                self.num_users = self.trainMatrix.shape[0]
                self.num_items = self.trainMatrix.shape[1]
                if os.path.exists(self.path+".negative"):
                    self.testNegatives = hold_out.load_negative_file()
                else :
                    self.testNegatives = self.get_negatives()
            else :
                self.trainMatrix,self.trainDict,self.validDict,self.validMatrix,self.testMatrix,\
                self.userseq,self.userids,self.itemids,self.timeMatrix =\
                hold_out.load_data_by_user_time()
                self.num_users = self.trainMatrix.shape[0]
                self.num_items = self.trainMatrix.shape[1]
                self.testNegatives = self.get_negatives()
            
        else :
            print("please choose a splitter")
        self.num_users = self.trainMatrix.shape[0]
        self.num_items = self.trainMatrix.shape[1]
               
    def get_negatives(self):
        negatives = {}
        for u in np.arange(self.num_users):
            negative_per_user =[]
            if(self.evaluate_neg>0):
                for _ in np.arange(self.evaluate_neg): #.....................
                    neg_item_id = np.random.randint(0,self.num_items)
                    while (u,neg_item_id) in self.trainMatrix.keys() or  (u,neg_item_id) in self.testMatrix.keys() \
                          or (u,neg_item_id) in self.validMatrix.keys() or neg_item_id in negative_per_user:
                        neg_item_id = np.random.randint(0, self.num_items)
                    negative_per_user.append(neg_item_id)
                negatives[u] = negative_per_user
                negative_per_user =[]
            else :
                negatives=None
        return  negatives                