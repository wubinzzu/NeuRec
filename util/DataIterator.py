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

    Arguments:
        data_source (_Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        super(SequentialSampler, self).__init__()
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (_Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        super(RandomSampler, self).__init__()
        self.data_source = data_source

    def __iter__(self):
        perm = np.random.permutation(len(self.data_source)).tolist()
        return iter(perm)

    def __len__(self):
        return len(self.data_source)


class BatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, sampler, batch_size, drop_last):
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
    """ pack the given data to one dataset

    Args:
        data: a list of 'data'.
    """
    def __init__(self, data):
        for d in data:
            assert len(d) == len(data[0])
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return [data[idx] for data in self.data]


class _DataLoaderIter(object):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""

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

    def __getstate__(self):
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("_DataLoaderIter cannot be pickled")

    def __del__(self):
        pass


class DataIterator(object):
    """
    Data loader. Combines a dataset and a sampler,
    and provides iterators over the dataset.

    Args:
        data: data from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, *data, batch_size=1, shuffle=False, drop_last=False):
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
    dataloader = DataIterator(users, items, batch_size=3, shuffle=False, drop_last=False)
    for bat_u, bat_i in dataloader:
        print(bat_u, bat_i)

    print()
    dataloader = DataIterator(users, items, batch_size=3, shuffle=True, drop_last=True)
    for bat_u, bat_i in dataloader:
        print(bat_u, bat_i)
