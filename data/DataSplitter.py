import scipy.sparse as sp
from util.Logger import logger
from util.Tool import get_data_format
import pandas as pd
import math


class DataSplitter(object):
    def __init__(self, path, splitter, data_format, separator, threshold, splitterRatio=[0.8,0.2]):
        self.path = path + ".rating"
        self.splitter = splitter
        self.separator = separator
        self.data_format = data_format
        self.splitterRatio = splitterRatio
        self.threshold = threshold
        if float(splitterRatio[0]) + float(splitterRatio[1]) != 1.0:
            raise ValueError("please given a correct splitterRatio")
        
    def load_data(self):
        logger.info("Loading interaction records from %s "%(self.path))
        time_matrix = None
        columns = get_data_format(self.data_format)
        data = pd.read_csv(self.path, sep=self.separator, header=None, names=columns)
        
        if self.data_format == "UIRT" or self.data_format == "UIR":
            data = data[data["rating"] >= self.threshold]

        unique_user = data["user"].unique()
        user2id = pd.Series(data=range(len(unique_user)), index=unique_user)
        data["user"] = data["user"].map(user2id)
        num_users = len(unique_user)
        userids = user2id.to_dict()

        unique_item = data["item"].unique()
        item2id = pd.Series(data=range(len(unique_item)), index=unique_item)
        data["item"] = data["item"].map(item2id)
        num_items = len(unique_item)
        itemids = item2id.to_dict()

        if self.data_format == "UIRT" or self.data_format == "UIT":
            data.sort_values(by=["user", "time"], inplace=True)
            time_matrix = sp.csr_matrix((data["time"], (data["user"], data["item"])), shape=(num_users, num_items))

        if self.splitter == "ratio":
            train_actions, test_actions = self.HoldOutDataSplitter(data)
            
        else:
            train_actions, test_actions = self.LeaveOneOutDataSplitter(data)
        
        if self.data_format == "UI": 
            train_matrix = sp.csr_matrix(([1]*len(train_actions["user"]), (train_actions["user"], train_actions["item"])), shape=(num_users, num_items))
            test_matrix = sp.csr_matrix(([1]*len(test_actions["user"]), (test_actions["user"], test_actions["item"])), shape=(num_users, num_items))      
        
        else:
            train_matrix = sp.csr_matrix((train_actions["rating"], (train_actions["user"], train_actions["item"])), shape=(num_users, num_items))
            test_matrix = sp.csr_matrix((test_actions["rating"], (test_actions["user"], test_actions["item"])), shape=(num_users, num_items))
            
        num_ratings = len(train_actions["user"]) + len(test_actions["user"])
        sparsity = 1- num_ratings/(num_users*num_items)
        logger.info("\"num_users\": %d,\"num_items\":%d, \"num_ratings\":%d, \"sparsity\":%.4f"%(num_users, num_items, num_ratings, sparsity))   
        return train_matrix, test_matrix, time_matrix, userids, itemids
    
    def HoldOutDataSplitter(self, data):
        train_actions = []
        test_actions = []

        user_grouped = data.groupby(by=["user"])
        for user, u_data in user_grouped:
            u_data_len = len(u_data)
            if self.data_format == "UIR" or self.data_format == "UI":
                u_data = u_data.sample(frac=1)
            idx = math.ceil(self.splitterRatio[0] * u_data_len)
            train_actions.append(u_data.iloc[:idx])
            test_actions.append(u_data.iloc[idx:])

        train_actions = pd.concat(train_actions, ignore_index=True)
        test_actions = pd.concat(test_actions, ignore_index=True)

        return train_actions, test_actions
    
    def LeaveOneOutDataSplitter(self, data):
        train_actions = []
        test_actions = []

        user_grouped = data.groupby(by=["user"])
        for user, u_data in user_grouped:
            u_data_len = len(u_data)
            if u_data_len <= 3:
                train_actions = train_actions.append(u_data)
            else:
                if self.data_format == "UIR" or self.data_format == "UI":
                    u_data = u_data.sample(frac=1)
                train_actions.append(u_data.iloc[:-1])
                test_actions.append(u_data.iloc[-1:])

        train_actions = pd.concat(train_actions, ignore_index=True)
        test_actions = pd.concat(test_actions, ignore_index=True)

        return train_actions, test_actions
