import scipy.sparse as sp
from util.tool import get_data_format
from util.logger import logger
import pandas as pd


class GivenData(object):
    def __init__(self, path, data_format, separator, threshold):
        self.path = path
        self.data_format = data_format
        self.separator = separator
        self.threshold = threshold

    def load_data(self):
        logger.info("Loading interaction records from %s "%(self.path))
        time_matrix = None
        columns = get_data_format(self.data_format)

        train_data = pd.read_csv(self.path+".train", sep=self.separator, header=None, names=columns)
        test_data = pd.read_csv(self.path + ".test", sep=self.separator, header=None, names=columns)
        
        if self.data_format == "UIRT" or self.data_format == "UIR":
            train_data = train_data[train_data["rating"] >= self.threshold]
            test_data = test_data[test_data["rating"] >= self.threshold]

        all_data = pd.concat([train_data, test_data])

        unique_user = all_data["user"].unique()
        unique_item = all_data["item"].unique()
        user2id = pd.Series(data=range(len(unique_user)), index=unique_user)
        item2id = pd.Series(data=range(len(unique_item)), index=unique_item)
        all_data["user"] = all_data["user"].map(user2id)
        train_data["user"] = train_data["user"].map(user2id)
        test_data["user"] = test_data["user"].map(user2id)

        all_data["item"] = all_data["item"].map(item2id)
        train_data["item"] = train_data["item"].map(item2id)
        test_data["item"] = test_data["item"].map(item2id)

        num_users = len(unique_user)
        num_items = len(unique_item)

        userids = user2id.to_dict()
        itemids = item2id.to_dict()
        
        if self.data_format == "UIRT" or self.data_format == "UIT":
            time_matrix = sp.csr_matrix((all_data["time"], (all_data["user"], all_data["item"])), shape=(num_users, num_items))  
        
        if self.data_format == "UI":
            train_matrix = sp.csr_matrix(([1]*len(train_data["user"]), (train_data["user"], train_data["item"])), shape=(num_users, num_items))
            test_matrix = sp.csr_matrix(([1]*len(test_data["user"]), (test_data["user"], test_data["item"])), shape=(num_users, num_items))  
            
        else: 
            train_matrix = sp.csr_matrix(([1]*len(train_data["rating"]), (train_data["user"], train_data["item"])), shape=(num_users, num_items))
            test_matrix = sp.csr_matrix(([1]*len(test_data["rating"]), (test_data["user"], test_data["item"])), shape=(num_users, num_items))  
        
        num_ratings = len(train_data["user"]) + len(test_data["user"])
        sparsity = 1 - num_ratings/(num_users*num_items)
        logger.info("\"num_users\": %d,\"num_items\":%d, \"num_ratings\":%d, \"sparsity\":%.4f"%(num_users, num_items, num_ratings, sparsity))
        return train_matrix, test_matrix, time_matrix, userids, itemids
