"""
@author: Zhongchuan Sun
"""
from util import DataIterator
from util.cython.random_choice import batch_randint_choice
from collections import Iterable
import numpy as np


class Sampler(object):
    """Base class for all sampler to sample negative items.
    """

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


def _generate_positive_items(user_pos_dict):
    if not isinstance(user_pos_dict, dict):
        raise TypeError("'user_pos_dict' must be a dict.")

    if not user_pos_dict:
        raise ValueError("'user_pos_dict' cannot be empty.")

    users_list, pos_items_list = [], []
    user_pos_len = []
    for user, pos_items in user_pos_dict.items():
        pos_len = len(pos_items)
        user_pos_len.append([user, pos_len])
        users_list.extend([user] * len(pos_items))
        pos_items_list.extend(pos_items)

    return user_pos_len, users_list, pos_items_list


def _generative_time_order_positive_items(user_pos_dict, high_order=1):
    if high_order <= 0:
        raise ValueError("'high_order' must be a positive integer.")

    if not isinstance(user_pos_dict, dict):
        raise TypeError("'user_pos_dict' must be a dict.")

    if not user_pos_dict:
        raise ValueError("'user_pos_dict' cannot be empty.")

    users_list, recent_items_list, pos_items_list = [], [], []
    user_pos_len = []
    for user, seq_items in user_pos_dict.items():
        if len(seq_items) - high_order <= 0:
            continue
        num_instance = len(seq_items) - high_order
        user_pos_len.append([user, num_instance])
        users_list.extend([user] * num_instance)
        if high_order == 1:
            r_items = [seq_items[idx] for idx in range(num_instance)]
        else:
            r_items = [seq_items[idx:][:high_order] for idx in range(num_instance)]

        recent_items_list.extend(r_items)
        pos_items_list.extend(seq_items[high_order:])

    return user_pos_len, users_list, recent_items_list, pos_items_list


def _sampling_negative_items(user_pos_len, neg_num, item_num, user_pos_dict):
    if neg_num <= 0:
        raise ValueError("'neg_num' must be a positive integer.")

    users, n_pos = list(zip(*user_pos_len))
    users_n_pos = DataIterator(users, n_pos, batch_size=1024, shuffle=False, drop_last=False)
    neg_items_list = []
    for bat_user, batch_num in users_n_pos:
        batch_num = [num * neg_num for num in batch_num]
        exclusion = [user_pos_dict[u] for u in bat_user]
        bat_neg_items = batch_randint_choice(item_num, batch_num, replace=True, exclusion=exclusion)

        for user, neg_items, n_item in zip(bat_user, bat_neg_items, batch_num):
            if isinstance(neg_items, Iterable):
                if neg_num > 1:
                    neg_items = np.reshape(neg_items, newshape=[-1, neg_num])
                neg_items_list.extend(neg_items)
            else:
                neg_items_list.append(neg_items)
    return neg_items_list


class PointwiseSampler(Sampler):
    """Sampling negative items and construct pointwise training instances.

    The training instances consist of `batch_user`, `batch_item` and
    `batch_label`, which are lists of users, items and labels. All lengths of
    them are `batch_size`.
    Positive and negative items are labeled as `1` and  `0`, respectively.
    """

    def __init__(self, dataset, neg_num=1, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `PointwiseSampler` instance.

        Args:
            dataset (data.Dataset): An instance of `Dataset`.
            neg_num (int): How many negative items for each positive item.
                Defaults to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(Sampler, self).__init__()
        if neg_num <= 0:
            raise ValueError("'neg_num' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.neg_num = neg_num
        self.item_num = dataset.num_items
        self.user_pos_dict = dataset.get_user_train_dict()
        self.user_pos_len, users_list, self.pos_items_list = \
            _generate_positive_items(self.user_pos_dict)

        self.users_list = users_list * (self.neg_num+1)
        len_pos_items = len(self.pos_items_list)
        pos_labels_list = [1.0] * len_pos_items
        neg_labels_list = [0.0] * (len_pos_items * self.neg_num)
        self.all_labels = pos_labels_list + neg_labels_list

    def __iter__(self):
        neg_items_list = _sampling_negative_items(self.user_pos_len, self.neg_num,
                                                  self.item_num, self.user_pos_dict)

        neg_items = np.array(neg_items_list, dtype=np.int32)
        neg_items = np.reshape(neg_items.T, [-1]).tolist()
        all_items = self.pos_items_list + neg_items

        data_iter = DataIterator(self.users_list, all_items, self.all_labels,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_items, bat_labels in data_iter:
            yield bat_users, bat_items, bat_labels

    def __len__(self):
        n_sample = len(self.users_list)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size


class PairwiseSampler(Sampler):
    """Sampling negative items and construct pairwise training instances.

    The training instances consist of `batch_user`, `batch_pos_item` and
    `batch_neg_items`, where `batch_user` and `batch_pos_item` are lists
    of users and positive items with length `batch_size`, and `neg_items`
    does not interact with `user`.

    If `neg_num == 1`, `batch_neg_items` is also a list of negative items
    with length `batch_size`;  If `neg_num > 1`, `batch_neg_items` is an
    array like list with shape `(batch_size, neg_num)`.
    """
    def __init__(self, dataset, neg_num=1, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `PairwiseSampler` instance.

        Args:
            dataset (data.Dataset): An instance of `Dataset`.
            neg_num (int): How many negative items for each positive item.
                Defaults to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(PairwiseSampler, self).__init__()
        if neg_num <= 0:
            raise ValueError("'neg_num' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.neg_num = neg_num
        self.item_num = dataset.num_items
        self.user_pos_dict = dataset.get_user_train_dict()

        self.user_pos_len, self.users_list, self.pos_items_list = \
            _generate_positive_items(self.user_pos_dict)

    def __iter__(self):
        neg_items_list = _sampling_negative_items(self.user_pos_len, self.neg_num,
                                                  self.item_num, self.user_pos_dict)

        data_iter = DataIterator(self.users_list, self.pos_items_list, neg_items_list,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle, drop_last=self.drop_last)
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            yield bat_users, bat_pos_items, bat_neg_items

    def __len__(self):
        n_sample = len(self.users_list)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size


class TimeOrderPointwiseSampler(Sampler):
    """Sampling negative items and construct time ordered pointwise instances.

    The training instances consist of `batch_user`, `batch_recent_items`,
    `batch_item` and `batch_label`. For each instance, positive `label`
    indicates that `user` interacts with `item` immediately following
    `recent_items`; and negative `label` indicates that `item` does not
    interact with `user`.

    If `high_order == 1`, `batch_recent_items` is a list of items with length
    `batch_size`; If `high_order > 1`, `batch_recent_items` is an array like
    list with shape `(batch_size, high_order)`.
    Positive and negative items are labeled as `1` and  `0`, respectively.
    """

    def __init__(self, dataset, high_order=1, neg_num=1, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `TimeOrderPointwiseSampler` instance.

        Args:
            dataset (data.Dataset): An instance of `Dataset`.
            high_order (int): The number of recent items. Defaults to `1`.
            neg_num (int): How many negative items for each positive item.
                Defaults to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(TimeOrderPointwiseSampler, self).__init__()
        if high_order < 0:
            raise ValueError("'high_order' must be a positive integer.")
        if neg_num <= 0:
            raise ValueError("'neg_num' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.neg_num = neg_num
        self.item_num = dataset.num_items
        self.user_pos_dict = dataset.get_user_train_dict(by_time=True)

        self.user_pos_len, users_list, recent_items_list, self.pos_items_list = \
            _generative_time_order_positive_items(self.user_pos_dict, high_order=high_order)

        self.users_list = users_list * (self.neg_num + 1)
        self.recent_items_list = recent_items_list * (self.neg_num + 1)

        len_pos_items = len(self.pos_items_list)
        pos_labels_list = [1.0] * len_pos_items
        neg_labels_list = [0.0] * (len_pos_items * self.neg_num)
        self.all_labels = pos_labels_list + neg_labels_list

    def __iter__(self):
        neg_items_list = _sampling_negative_items(self.user_pos_len, self.neg_num,
                                                  self.item_num, self.user_pos_dict)

        neg_items = np.array(neg_items_list, dtype=np.int32)
        neg_items = np.reshape(neg_items.T, [-1]).tolist()
        all_next_items = self.pos_items_list + neg_items

        data_iter = DataIterator(self.users_list, self.recent_items_list, all_next_items, self.all_labels,
                                 batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_recent_items, bat_next_items, bat_labels in data_iter:
            yield bat_users, bat_recent_items, bat_next_items, bat_labels

    def __len__(self):
        n_sample = len(self.users_list)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size


class TimeOrderPairwiseSampler(Sampler):
    """Sampling negative items and construct time ordered pairwise instances.

    The training instances consist of `batch_user`, `batch_recent_items`,
    `batch_next_item` and `batch_neg_items`. For each instance, `user`
    interacts with `next_item` immediately following `recent_items`, and
    `neg_items` does not interact with `user`.

    If `high_order == 1`, `batch_recent_items` is a list of items with length
    `batch_size`; If `high_order > 1`, `batch_recent_items` is an array like
    list with shape `(batch_size, high_order)`.

    If `neg_num == 1`, `batch_neg_items` is a list of negative items with length
    `batch_size`; If `neg_num > 1`, `batch_neg_items` is an array like list with
    shape `(batch_size, neg_num)`.
    """
    def __init__(self, dataset, high_order=1, neg_num=1, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `TimeOrderPairwiseSampler` instance.

        Args:
            dataset (data.Dataset): An instance of `Dataset`.
            high_order (int): The number of recent items. Defaults to `1`.
            neg_num (int): How many negative items for each positive item.
                Defaults to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(TimeOrderPairwiseSampler, self).__init__()
        if high_order < 0:
            raise ValueError("'high_order' must be a positive integer.")
        if neg_num <= 0:
            raise ValueError("'neg_num' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.neg_num = neg_num
        self.item_num = dataset.num_items
        self.user_pos_dict = dataset.get_user_train_dict(by_time=True)

        self.user_pos_len, self.users_list, self.recent_items_list, self.pos_items_list = \
            _generative_time_order_positive_items(self.user_pos_dict, high_order=high_order)

    def __iter__(self):
        neg_items_list = _sampling_negative_items(self.user_pos_len, self.neg_num,
                                                  self.item_num, self.user_pos_dict)

        data_iter = DataIterator(self.users_list, self.recent_items_list, self.pos_items_list, neg_items_list,
                                 batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_recent_items, bat_pos_items, bat_neg_items in data_iter:
            yield bat_users, bat_recent_items, bat_pos_items, bat_neg_items

    def __len__(self):
        n_sample = len(self.users_list)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size
