"""
@author: Zhongchuan Sun
"""
from multiprocessing import JoinableQueue, Process
from util import DataIterator
import random
from multiprocessing.queues import Empty
import traceback
import sys


class ParallelSampler(object):
    """Base class for all parallel sampler to sample negative items.

    This class aims to parallelize the training and sampling processes by
    producer-consumer model.

    The parallelization and sample iteration are already implemented in
    this abstract class. Every subclass has to provide the no argument
    `sampling` method.
    """

    def __init__(self, batch_size=1, drop_last=False):
        """Initializes a new `ParallelSampler` instance.

        Args:
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.queue = JoinableQueue(maxsize=10 * self.batch_size)

    def _sampling_wrap(self):
        """Capture the exceptions in `sampling` and put it into `queue`.
        """
        try:
            self.sampling()
        except BaseException as e:
            tb = traceback.format_exc()
            self.queue.put((e, tb))

    def sampling(self):
        """Sample training instances.

        The constructed training instances need to put into the attribute
        `queue` to help the parallelization of training and sampling processes.

        Example:
            There is a simple implementation of the `sampling` function::

                def sampling():
                    for i,j in enumerate(range(100, 120)):
                        self.queue.put([i,j])

            And then, `ParallelSampler` can be used in the following way::

                data_sampler = ParallelSampler(batch_size=4)
                for i, j in data_sampler:
                    print(i, j)

        """
        raise NotImplementedError

    def _next_batch_data(self):
        bat_data = []
        try:
            for _ in range(self.batch_size):
                data = self.queue.get(timeout=5)
                # Detect and print exception in `sampling`, and then exit.
                if len(data) == 2 and isinstance(data[0], BaseException):
                    sys.stderr.write(data[1])
                    exit(1)
                bat_data.append(data)
                self.queue.task_done()
        except Empty:
            bat_len = len(bat_data)
            if (bat_len == 0) or (bat_len < self.batch_size and self.drop_last):
                raise StopIteration

        return list(zip(*bat_data))

    def __iter__(self):
        sampler = Process(target=self._sampling_wrap, args=())
        sampler.daemon = True
        sampler.start()

        try:
            while True:
                yield self._next_batch_data()
        except StopIteration:
            sampler.join()
            sampler.terminate()
            raise StopIteration


class PointwiseSampler(ParallelSampler):
    """Sampling negative items and construct pointwise training instances.

    The training instances consist of `batch_user`, `batch_item` and
    `batch_label`, which are lists of users, items and labels. All lengths of
    them are `batch_size`.
    Positive and negative items are labeled as `1` and  `0`, respectively.
    """

    def __init__(self, dataset, neg_num=1, batch_size=1, shuffle=False, drop_last=False):
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
        super(PointwiseSampler, self).__init__(batch_size=batch_size, drop_last=drop_last)
        self.shuffle = shuffle
        self.neg_num = neg_num
        self.item_unm = dataset.num_items
        user_pos_dict = dataset.get_user_train_dict()
        self.user_pos_dict = {user: set(items) for user, items in user_pos_dict.items()}
        user_list, item_list = dataset.get_train_interactions()
        self.ui_interactions = DataIterator(user_list, item_list, batch_size=1,
                                            shuffle=self.shuffle, drop_last=False)

    def sampling(self):
        """Sample negative items and construct training instances.

        A sample implementation of this function is shown in
        `ParallelSampler.sampling`.
        """
        for user, pos_item in self.ui_interactions:
            user, pos_item = user[0], pos_item[0]
            self.queue.put([user, pos_item, 1])
            for _ in range(self.neg_num):
                while True:
                    neg_item = random.randint(0, self.item_unm - 1)
                    if neg_item not in self.user_pos_dict[user]:
                        self.queue.put([user, neg_item, 0])
                        break

    def __len__(self):
        pos_len = len(self.ui_interactions)
        return (1 + self.neg_num) * pos_len


class PairwiseSampler(ParallelSampler):
    """Sampling negative items and construct pairwise training instances.

    The training instances consist of `batch_user`, `batch_pos_item` and
    `batch_neg_items`, where `batch_user` and `batch_pos_item` are lists
    of users and positive items with length `batch_size`, and `neg_items`
    does not interact with `user`.

    If `neg_num == 1`, `batch_neg_items` is also a list of negative items
    with length `batch_size`;  If `neg_num > 1`, `batch_neg_items` is an
    array like list with shape `(batch_size, neg_num)`.
    """

    def __init__(self, dataset, neg_num=1, batch_size=1, shuffle=False, drop_last=False):
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
        super(PairwiseSampler, self).__init__(batch_size=batch_size, drop_last=drop_last)
        self.shuffle = shuffle
        self.neg_num = neg_num
        self.item_unm = dataset.num_items
        user_pos_dict = dataset.get_user_train_dict()
        self.user_pos_dict = {user: set(items) for user, items in user_pos_dict.items()}
        user_list, item_list = dataset.get_train_interactions()
        self.ui_interactions = DataIterator(user_list, item_list, batch_size=1,
                                            shuffle=self.shuffle, drop_last=False)

    def sampling(self):
        """Sample negative items and construct training instances.

        A sample implementation of this function is shown in
        `ParallelSampler.sampling`.
        """
        for user, pos_item in self.ui_interactions:
            user, pos_item = user[0], pos_item[0]
            neg_items = []
            for _ in range(self.neg_num):
                while True:
                    neg_item = random.randint(0, self.item_unm - 1)
                    if neg_item not in self.user_pos_dict[user]:
                        neg_items.append(neg_item)
                        break
            neg_items = neg_items if self.neg_num > 1 else neg_items[0]
            self.queue.put([user, pos_item, neg_items])

    def __len__(self):
        pos_len = len(self.ui_interactions)
        return pos_len


class TimeOrderPointwiseSampler(ParallelSampler):
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

    def __init__(self, dataset, high_order=1, neg_num=1, batch_size=1, shuffle=False, drop_last=False):
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
        super(TimeOrderPointwiseSampler, self).__init__(batch_size=batch_size, drop_last=drop_last)
        if high_order < 0:
            raise ValueError("'high_order' can be a negative integer!")

        self.shuffle = shuffle
        self.neg_num = neg_num
        self.item_unm = dataset.num_items
        user_pos_dict = dataset.get_user_train_dict(by_time=True)
        self.user_pos_dict = {user: set(items) for user, items in user_pos_dict.items()}

        user_list, recent_items_list, next_item_list = [], [], []
        for user, seq_items in user_pos_dict.items():
            num_instance = len(seq_items) - high_order
            user_list.extend([user] * num_instance)
            if high_order == 1:
                r_items = [seq_items[idx] for idx in range(num_instance)]
            else:
                r_items = [seq_items[idx:][:high_order] for idx in range(num_instance)]

            recent_items_list.extend(r_items)
            next_item_list.extend(seq_items[high_order:])

        self.ui_interactions = DataIterator(user_list, recent_items_list, next_item_list,
                                            batch_size=1, shuffle=self.shuffle, drop_last=False)

    def sampling(self):
        """Sample negative items and construct training instances.

        A sample implementation of this function is shown in
        `ParallelSampler.sampling`.
        """
        for user, recent_items, next_item in self.ui_interactions:
            user, recent_items, next_item = user[0], recent_items[0], next_item[0]
            self.queue.put([user, recent_items, next_item, 1])

            for _ in range(self.neg_num):
                while True:
                    neg_item = random.randint(0, self.item_unm - 1)
                    if neg_item not in self.user_pos_dict[user]:
                        self.queue.put([user, recent_items, neg_item, 0])
                        break

    def __len__(self):
        pos_len = len(self.ui_interactions)
        return (1 + self.neg_num) * pos_len


class TimeOrderPairwiseSampler(ParallelSampler):
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

    def __init__(self, dataset, high_order=1, neg_num=1, batch_size=1, shuffle=False, drop_last=False):
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
        super(TimeOrderPairwiseSampler, self).__init__(batch_size=batch_size, drop_last=drop_last)
        if high_order < 0:
            raise ValueError("'high_order' can be a negative integer!")

        self.shuffle = shuffle
        self.neg_num = neg_num
        self.item_unm = dataset.num_items
        user_pos_dict = dataset.get_user_train_dict(by_time=True)
        self.user_pos_dict = {user: set(items) for user, items in user_pos_dict.items()}

        user_list, recent_items_list, next_item_list = [], [], []
        for user, seq_items in user_pos_dict.items():
            num_instance = len(seq_items) - high_order
            user_list.extend([user] * num_instance)
            if high_order == 1:
                r_items = [seq_items[idx] for idx in range(num_instance)]
            else:
                r_items = [seq_items[idx:][:high_order] for idx in range(num_instance)]

            recent_items_list.extend(r_items)
            next_item_list.extend(seq_items[high_order:])

        self.ui_interactions = DataIterator(user_list, recent_items_list, next_item_list,
                                            batch_size=1, shuffle=self.shuffle, drop_last=False)

    def sampling(self):
        """Sample negative items and construct training instances.

        A sample implementation of this function is shown in
        `ParallelSampler.sampling`.
        """
        for user, recent_item, next_item in self.ui_interactions:
            user, recent_item, next_item = user[0], recent_item[0], next_item[0]
            neg_items = []
            for _ in range(self.neg_num):
                while True:
                    neg_item = random.randint(0, self.item_unm - 1)
                    if neg_item not in self.user_pos_dict[user]:
                        neg_items.append(neg_item)
                        break

            neg_items = neg_items if self.neg_num > 1 else neg_items[0]
            self.queue.put([user, recent_item, next_item, neg_items])

    def __len__(self):
        pos_len = len(self.ui_interactions)
        return pos_len
