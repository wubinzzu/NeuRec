__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Dataset", "Interaction"]

import scipy.sparse as sp
import os
import warnings
import pandas as pd
import numpy as np
from reckit import typeassert
from collections import OrderedDict
from copy import deepcopy
from reckit import pad_sequences

_USER = "user"
_ITEM = "item"
_RATING = "rating"
_TIME = "time"
_column_dict = {"UI": [_USER, _ITEM],
                "UIR": [_USER, _ITEM, _RATING],
                "UIT": [_USER, _ITEM, _TIME],
                "UIRT": [_USER, _ITEM, _RATING, _TIME]
                }


class Interaction(object):
    @typeassert(data=(pd.DataFrame, None), num_users=(int, None), num_items=(int, None))
    def __init__(self, data=None, num_users=None, num_items=None):
        if data is None or data.empty:
            self._data = pd.DataFrame()
            self.num_users = 0
            self.num_items = 0
            self.num_ratings = 0
        else:
            self._data = data
            self.num_users = num_users if num_users is not None else max(data[_USER]) + 1
            self.num_items = num_items if num_items is not None else max(data[_ITEM]) + 1
            self.num_ratings = len(data)

        self._buffer = dict()

    def to_user_item_pairs(self):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None
        # users_np = self._data[_USER].to_numpy(copy=True, dtype=np.int32)
        # items_np = self._data[_ITEM].to_numpy(copy=True, dtype=np.int32)
        ui_pairs = self._data[[_USER, _ITEM]].to_numpy(copy=True, dtype=np.int32)
        return ui_pairs

    def to_csr_matrix(self):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None
        users, items = self._data[_USER].to_numpy(), self._data[_ITEM].to_numpy()
        ratings = self._data[_RATING].to_numpy() if _RATING in self._data else np.zeros(len(users), dtype=np.float32)
        csr_mat = sp.csr_matrix((ratings, (users, items)), shape=(self.num_users, self.num_items))
        return csr_mat

    def to_dok_matrix(self):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None
        return self.to_csr_matrix().todok()

    def to_coo_matrix(self):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None
        return self.to_csr_matrix().tocoo()

    def to_user_dict(self, by_time=False):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None

        if by_time and _TIME not in self._data:
            raise ValueError("This dataset do not have timestamp.")

        # read from buffer
        if by_time is True and "user_dict_byt" in self._buffer:
            return deepcopy(self._buffer["user_dict_byt"])
        if by_time is False and "user_dict" in self._buffer:
            return deepcopy(self._buffer["user_dict"])

        user_dict = OrderedDict()
        user_grouped = self._data.groupby(_USER)
        for user, user_data in user_grouped:
            if by_time:
                user_data = user_data.sort_values(by=[_TIME])
            user_dict[user] = user_data[_ITEM].to_numpy(dtype=np.int32)

        # write to buffer
        if by_time is True:
            self._buffer["user_dict_byt"] = deepcopy(user_dict)
        else:
            self._buffer["user_dict"] = deepcopy(user_dict)
        return user_dict

    def to_truncated_seq_dict(self, max_len, pad_value=0, padding='post', truncating='post'):
        """Get the truncated item sequences of each user.

        Args:
            max_len (int or None): Maximum length of all sequences.
            pad_value: Padding value. Defaults to `0.`.
            padding (str): `"pre"` or `"post"`: pad either before or after each
                sequence. Defaults to `post`.
            truncating (str): `"pre"` or `"post"`: remove values from sequences
                larger than `max_len`, either at the beginning or at the end of
                the sequences. Defaults to `post`.

        Returns:
            OrderedDict: key is user and value is truncated item sequences.

        """
        user_seq_dict = self.to_user_dict(by_time=True)
        if max_len is None:
            max_len = max([len(seqs) for seqs in user_seq_dict.values()])
        item_seq_list = [item_seq[-max_len:] for item_seq in user_seq_dict.values()]
        item_seq_arr = pad_sequences(item_seq_list, value=pad_value, max_len=max_len,
                                     padding=padding, truncating=truncating, dtype=np.int32)

        seq_dict = OrderedDict([(user, item_seq) for user, item_seq in
                                zip(user_seq_dict.keys(), item_seq_arr)])
        return seq_dict

    def _clean_buffer(self):
        self._buffer.clear()

    def update(self, other):
        """Update this object with the union of itself and other.
        Args:
            other (Interaction): An object of Interaction

        """
        if not isinstance(other, Interaction):
            raise TypeError("'other' must be a object of 'Interaction'")
        other_data = other._data
        if other_data.empty:
            warnings.warn("'other' is empty and update nothing.")
        elif self._data.empty:
            self._data = other_data.copy()
            self.num_users = other.num_users
            self.num_items = other.num_items
            self.num_ratings = other.num_items
            self._clean_buffer()
        elif self._data is other_data:
            warnings.warn("'other' is equal with self and update nothing.")
        else:
            self._data = pd.concat([self._data, other_data])
            self._data.drop_duplicates(inplace=True)
            self.num_users = max(self._data[_USER]) + 1
            self.num_items = max(self._data[_ITEM]) + 1
            self.num_ratings = len(self._data)
            self._clean_buffer()

    def union(self, other):
        """Return the union of self and other as a new Interaction.

        Args:
            other (Interaction): An object of Interaction.

        Returns:
            Interaction: The union of self and other.

        """
        if not isinstance(other, Interaction):
            raise TypeError("'other' must be a object of 'Interaction'")
        result = Interaction()
        result.update(self)
        result.update(other)
        return result

    def __add__(self, other):
        return self.union(other)

    def __bool__(self):
        return self.__len__() > 0

    def __len__(self):
        return len(self._data)


class Dataset(object):
    def __init__(self, data_dir, sep, columns):
        """Dataset

        Notes:
            The prefix name of data files is same as the data_dir, and the
            suffix/extension names are 'train', 'test', 'user2id', 'item2id'.
            Directory structure:
                data_dir
                    ├── data_dir.train      // training data
                    ├── data_dir.valid      // validation data, optional
                    ├── data_dir.test       // test data
                    ├── data_dir.user2id    // user to id, optional
                    ├── data_dir.item2id    // item to id, optional

        Args:
            data_dir: The directory of dataset.
            sep: The separator/delimiter of file columns.
            columns: The format of columns, must be one of 'UI',
                'UIR', 'UIT' and 'UIRT'
        """

        self._data_dir = data_dir
        self.data_name = os.path.split(data_dir)[-1]

        # metadata
        self.train_data = Interaction()
        self.valid_data = Interaction()
        self.test_data = Interaction()
        self.user2id = None
        self.item2id = None
        self.id2user = None
        self.id2item = None

        # statistic
        self.num_users = 0
        self.num_items = 0
        self.num_ratings = 0
        self._load_data(data_dir, sep, columns)

    def _load_data(self, data_dir, sep, columns):
        if columns not in _column_dict:
            key_str = ", ".join(_column_dict.keys())
            raise ValueError("'columns' must be one of '%s'." % key_str)

        columns = _column_dict[columns]

        file_prefix = os.path.join(data_dir, os.path.split(data_dir)[-1])

        # load data
        train_file = file_prefix+".train"
        if os.path.isfile(train_file):
            _train_data = pd.read_csv(train_file, sep=sep, header=None, names=columns)
        else:
            raise FileNotFoundError("%s does not exist." % train_file)

        valid_file = file_prefix + ".valid"
        if os.path.isfile(valid_file):
            _valid_data = pd.read_csv(valid_file, sep=sep, header=None, names=columns)
        else:
            _valid_data = pd.DataFrame()
            warnings.warn("%s does not exist." % valid_file)

        test_file = file_prefix + ".test"
        if os.path.isfile(test_file):
            _test_data = pd.read_csv(test_file, sep=sep, header=None, names=columns)
        else:
            raise FileNotFoundError("%s does not exist." % test_file)

        user2id_file = file_prefix + ".user2id"
        if os.path.isfile(user2id_file):
            _user2id = pd.read_csv(user2id_file, sep=sep, header=None).to_numpy()
            self.user2id = OrderedDict(_user2id)
            self.id2user = OrderedDict([(idx, user) for user, idx in self.user2id.items()])
        else:
            warnings.warn("%s does not exist." % user2id_file)

        item2id_file = file_prefix + ".item2id"
        if os.path.isfile(item2id_file):
            _item2id = pd.read_csv(item2id_file, sep=sep, header=None).to_numpy()
            self.item2id = OrderedDict(_item2id)
            self.id2item = OrderedDict([(idx, item) for item, idx in self.item2id.items()])
        else:
            warnings.warn("%s does not exist." % item2id_file)

        # statistical information
        data_list = [data for data in [_train_data, _valid_data, _test_data] if not data.empty]
        all_data = pd.concat(data_list)
        self.num_users = max(all_data[_USER]) + 1
        self.num_items = max(all_data[_ITEM]) + 1
        self.num_ratings = len(all_data)

        # convert to to the object of Interaction
        self.train_data = Interaction(_train_data, num_users=self.num_users, num_items=self.num_items)
        self.valid_data = Interaction(_valid_data, num_users=self.num_users, num_items=self.num_items)
        self.test_data = Interaction(_test_data, num_users=self.num_users, num_items=self.num_items)

    def __str__(self):
        """The statistic of dataset.

        Returns:
            str: The summary of statistic
        """
        if 0 in {self.num_users, self.num_items, self.num_ratings}:
            return "statistical information is unavailable now"
        else:
            num_users, num_items = self.num_users, self.num_items
            num_ratings = self.num_ratings
            sparsity = 1 - 1.0 * num_ratings / (num_users * num_items)

            statistic = ["Dataset statistics:",
                         "Name: %s" % self.data_name,
                         "The number of users: %d" % num_users,
                         "The number of items: %d" % num_items,
                         "The number of ratings: %d" % num_ratings,
                         "Average actions of users: %.2f" % (1.0 * num_ratings / num_users),
                         "Average actions of items: %.2f" % (1.0 * num_ratings / num_items),
                         "The sparsity of the dataset: %.6f%%" % (sparsity * 100),
                         "",
                         "The number of training: %d" % len(self.train_data),
                         "The number of validation: %d" % len(self.valid_data),
                         "The number of testing: %d" % len(self.test_data)
                         ]
            statistic = "\n".join(statistic)
            return statistic

    def __repr__(self):
        return self.__str__()
