"""
@author: Zhongchuan Sun
"""
from .utils import load_data, filter_data, remap_id
from .utils import split_by_ratio, split_by_loo
from util.logger import Logger
from util import randint_choice
import pandas as pd
import numpy as np
import os


class Splitter(object):
    def __init__(self, config):
        self.filename = os.path.join(config["data.input.path"], config["data.input.dataset"])
        self.filename += ".rating"
        self.ratio = config["ratio"]
        self.file_format = config["data.column.format"]
        self.sep = config["data.convert.separator"]
        self.user_min = config["user_min"]
        self.item_min = config["item_min"]
        self.by_time = config["by_time"]
        self.splitter = config["splitter"]
        self.test_neg = config["rec.evaluate.neg"]

    def split(self):
        if self.file_format.lower() == "uirt":
            columns = ["user", "item", "rating", "time"]
            by_time = self.by_time
        elif self.file_format.lower() == "uir":
            columns = ["user", "item", "rating"]
            by_time = False
        elif self.file_format.lower() == "ui":
            columns = ["user", "item"]
            by_time = False
        else:
            raise ValueError("'%s' is an invalid data column format!" % self.file_format)

        print("load data...")
        all_data = load_data(self.filename, sep=self.sep, columns=columns)
        print("filter data...")
        filtered_data = filter_data(all_data, user_min=self.user_min, item_min=self.item_min)
        print("remap id...")
        remapped_data, user2id, item2id = remap_id(filtered_data)

        user_num = len(remapped_data["user"].unique())
        item_num = len(remapped_data["item"].unique())
        rating_num = len(remapped_data["item"])
        sparsity = 1 - 1.0 * rating_num / (user_num * item_num)
        # sampling negative item for test
        if self.test_neg > 0:
            neg_items = []
            grouped_user = remapped_data.groupby(["user"])
            for user, u_data in grouped_user:
                line = [user]
                line.extend(randint_choice(item_num, size=self.test_neg, replace=False, exclusion=u_data["item"]))
                neg_items.append(line)

            neg_items = pd.DataFrame(neg_items)
        else:
            neg_items = None

        print("split data...")
        if self.splitter == "ratio":
            train_data, test_data = split_by_ratio(remapped_data, ratio=self.ratio, by_time=by_time)
        elif self.splitter == "loo":
            train_data, test_data = split_by_loo(remapped_data, by_time=by_time)
        else:
            raise ValueError("There is not splitter '%s'" % self.splitter)

        print("save to file...")
        base_name = os.path.basename(self.filename).split(".")[0]
        dir_name = os.path.dirname(self.filename)
        dir_name = os.path.join(dir_name, base_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        filename = "%s_%s_u%d_i%d" % (base_name, self.splitter, self.user_min, self.item_min)

        filename = os.path.join(dir_name, filename)
        np.savetxt(filename + ".train", train_data, fmt='%d', delimiter=self.sep)
        np.savetxt(filename + ".test", test_data, fmt='%d', delimiter=self.sep)

        # train_data.to_csv(filename + ".train", header=False, index=False, sep=self.sep)
        # test_data.to_csv(filename + ".test", header=False, index=False, sep=self.sep)
        if neg_items is not None:
            np.savetxt(filename + ".neg", neg_items, fmt='%d', delimiter=self.sep)
            # neg_items.to_csv(filename + ".neg", header=False, index=False, sep=self.sep)

        user2id = [[user, id] for user, id in user2id.to_dict().items()]
        item2id = [[item, id] for item, id in item2id.to_dict().items()]
        np.savetxt(filename + ".user2id", user2id, fmt='%s', delimiter=self.sep)
        np.savetxt(filename + ".item2id", item2id, fmt='%s', delimiter=self.sep)
        # user2id.to_csv(filename+".user2id", header=False, index=True, sep=self.sep)
        # item2id.to_csv(filename + ".item2id", header=False, index=True, sep=self.sep)

        logger = Logger(filename+".info")
        logger.info(self.filename)
        logger.info("The number of users: %d" % user_num)
        logger.info("The number of items: %d" % item_num)
        logger.info("The number of ratings: %d" % rating_num)
        logger.info("Average actions of users: %.2f" % (1.0*rating_num/user_num))
        logger.info("Average actions of items: %.2f" % (1.0*rating_num/item_num))
        logger.info("The sparsity of the dataset: %f%%" % (sparsity*100))
