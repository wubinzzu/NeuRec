"""
@author: Zhongchuan Sun
"""
from .utils import load_data, filter_data, remap_id
from .utils import split_by_ratio, split_by_loo
from util.Logger import Logger
import os


class Splitter(object):
    def __init__(self, config):
        self.filename = config["data_file"]
        self.ratio = eval(config["ratio"])
        self.file_format = config["file_format"]
        self.sep = eval(config["separator"])
        self.user_min = eval(config["user_min"])
        self.item_min = eval(config["item_min"])
        self.by_time = eval(config["by_time"])
        self.spliter = config["splitter"]

    def split(self):
        if self.file_format.lower() == "uirt":
            columns = ["user", "item", "rating", "time"]
            if self.by_time is False:
                by_time = False
            else:
                by_time = True
        elif self.file_format.lower() == "uir":
            columns = ["user", "item", "rating"]
            by_time = False
        else:
            raise ValueError("There is not data format '%s'" % self.file_format)

        print("load data...")
        all_data = load_data(self.filename, sep=self.sep, columns=columns)
        print("filter data...")
        filtered_data = filter_data(all_data, user_min=self.user_min, item_min=self.item_min)
        print("remap id...")
        remapped_data, user2id, item2id = remap_id(filtered_data)

        print("split data...")
        if self.spliter == "ratio":
            train_data, test_data = split_by_ratio(remapped_data, ratio=self.ratio, by_time=by_time)
        elif self.spliter == "loo":
            train_data, test_data = split_by_loo(remapped_data, by_time=by_time)
        else:
            raise ValueError("There is not splitter '%s'" % self.spliter)

        print("save to file...")
        base_name = os.path.basename(self.filename).split(".")[0]
        dir_name = os.path.dirname(self.filename)
        dir_name = os.path.join(dir_name, base_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        filename = "%s_%s_u%d_i%d" % (base_name, self.spliter, self.user_min, self.item_min)

        filename = os.path.join(dir_name, filename)
        train_data.to_csv(filename+".train", header=False, index=False)
        test_data.to_csv(filename + ".test", header=False, index=False)

        user2id.to_csv(filename+".user2id", header=False, index=True)
        item2id.to_csv(filename + ".item2id", header=False, index=True)

        user_num = len(remapped_data["user"].unique())
        item_num = len(remapped_data["item"].unique())
        rating_num = len(remapped_data["item"])
        sparsity = 1-1.0*rating_num/(user_num*item_num)
        logger = Logger(filename+".info")
        logger.info(self.filename)
        logger.info("The number of users: %d" % user_num)
        logger.info("The number of items: %d" % item_num)
        logger.info("The number of ratings: %d" % rating_num)
        logger.info("Average actions of users: %.2f" % (1.0*rating_num/user_num))
        logger.info("Average actions of items: %.2f" % (1.0*rating_num/item_num))
        logger.info("The sparsity of the dataset: %f%%" % (sparsity*100))
