"""
Created on Aug 8, 2016
Processing datasets. 
@author: Xiangnan He (xiangnanhe@gmail.com)
"""

import pandas as pd
from scipy.sparse import csr_matrix


class Dataset(object):
    def __init__(self, conf):
        """
        Constructor
        """
        self.train_matrix = None
        self.test_matrix = None
        self.time_matrix = None
        # self.userids = None
        # self.itemids = None
        self.num_users = None
        self.num_items = None
        self.dataset_name = conf["data.input.dataset"]

        self._load_data(conf)

    def _load_data(self, config):
        path = config["data.input.path"]
        dataset_name = config["data.input.dataset"]
        path = path + dataset_name

        train_file = path + ".train"
        test_file = path + ".test"

        file_format = config["data.column.format"]
        sep = config["data.convert.separator"]

        if file_format == "UIRT":
            columns = ["user", "item", "rating", "time"]
        elif file_format == "UIR":
            columns = ["user", "item", "rating"]
        elif file_format == "UI":
            columns = ["user", "item"]
        else:
            raise ValueError("There is not data format '%s'" % file_format)

        train_data = pd.read_csv(train_file, sep=sep, header=None, names=columns)
        test_data = pd.read_csv(test_file, sep=sep, header=None, names=columns)

        all_data = pd.concat([train_data, test_data])

        self.num_users = len(all_data["user"].unique())
        self.num_items = len(all_data["item"].unique())
        self.num_ratings = len(all_data)

        if file_format == "UI":
            self.train_matrix = csr_matrix(([1.0]*len(train_data["user"]), (train_data["user"], train_data["item"])),
                                           shape=(self.num_users, self.num_items))
            self.test_matrix = csr_matrix(([1.0]*test_data["user"], (test_data["user"], test_data["item"])),
                                          shape=(self.num_users, self.num_items))
        else:
            self.train_matrix = csr_matrix((train_data["rating"], (train_data["user"], train_data["item"])),
                                           shape=(self.num_users, self.num_items))
            self.test_matrix = csr_matrix((test_data["rating"], (test_data["user"], test_data["item"])),
                                          shape=(self.num_users, self.num_items))
        if file_format == "UIRT":
            self.time_matrix = csr_matrix((train_data["time"], (train_data["user"], train_data["item"])),
                                          shape=(self.num_users, self.num_items))
