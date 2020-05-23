"""
@author: Zhongchuan Sun
"""
import numpy as np


class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.
    """

    def __init__(self, data_source):
        """Initializes a new `SequentialSampler` instance.

        Args:
            data_source (_Dataset): Dataset to sample from.
        """
        super(SequentialSampler, self).__init__()
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.
    """

    def __init__(self, data_source):
        """Initializes a new `SequentialSampler` instance.

        Args:
            data_source (_Dataset): Dataset to sample from.
        """
        super(RandomSampler, self).__init__()
        self.data_source = data_source

    def __iter__(self):
        perm = np.random.permutation(len(self.data_source)).tolist()
        return iter(perm)

    def __len__(self):
        return len(self.data_source)


class BatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.
    """

    def __init__(self, sampler, batch_size, drop_last):
        """Initializes a new `BatchSampler` instance.

        Args:
            sampler (Sampler): Base sampler.
            batch_size (int): Size of mini-batch.
            drop_last (bool): If `True`, the sampler will drop the last batch
                if its size would be less than `batch_size`.
        """
        super(BatchSampler, self).__init__()
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class _Dataset(object):
    """Pack the given data to one dataset.

    Args:
        data (list or tuple): a list of 'data'.
    """

    def __init__(self, data):
        for d in data:
            if len(d) != len(data[0]):
                raise ValueError("The length of the given data are not equal!")
            # assert len(d) == len(data[0])
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return [data[idx] for data in self.data]


class _DataLoaderIter(object):
    """Iterates once over the dataset, as specified by the sampler.
    """

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_sampler = loader.batch_sampler
        self.sample_iter = iter(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        indices = next(self.sample_iter)  # may raise StopIteration
        batch = [self.dataset[i] for i in indices]

        transposed = [list(samples) for samples in zip(*batch)]
        if len(transposed) == 1:
            transposed = transposed[0]
        return transposed

    def __iter__(self):
        return self


class DataIterator(object):
    """`DataIterator` provides iterators over the dataset.

    This class combines some data sets and provides a batch iterator over them.
    For example::

        users = list(range(10))
        items = list(range(10, 20))
        labels = list(range(20, 30))

        data_iter = DataIterator(users, items, labels, batch_size=4, shuffle=False)
        for bat_user, bat_item, bat_label in data_iter:
            print(bat_user, bat_item, bat_label)

        data_iter = DataIterator(users, items, batch_size=4, shuffle=True, drop_last=True)
        for bat_user, bat_item in data_iter:
            print(bat_user, bat_item)

    """

    def __init__(self, *data, batch_size=1, shuffle=False, drop_last=False):
        """
        Args:
            *data: Variable length data list.
            batch_size (int): How many samples per batch to load. Defaults to `1`.
            shuffle (bool): Set to `True` to have the data reshuffled at every
                epoch. Defaults to `False`.
            drop_last (bool): Set to `True` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size.
                If `False` and the size of dataset is not divisible by the
                batch size, then the last batch will be smaller.
                Defaults to `False`.

        Raises:
            ValueError: If the length of the given data are not equal.
        """
        dataset = _Dataset(list(data))
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)


if __name__ == "__main__":
    users = list(range(10))
    items = list(range(10, 20))
    labels = list(range(20, 30))

    data_iter = DataIterator(users, items, labels, batch_size=4, shuffle=False)
    for bat_user, bat_item, bat_label in data_iter:
        print(bat_user, bat_item, bat_label)

    data_iter = DataIterator(users, items, batch_size=4, shuffle=True, drop_last=True)
    for bat_user, bat_item in data_iter:
        print(bat_user, bat_item)
