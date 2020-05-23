"""
@author: Zhongchuan Sun
"""
import pandas as pd
import math
import hashlib
import os


def check_md5(file_name):
    if not os.path.isfile(file_name):
        raise FileNotFoundError("There is not file named '%s'!" % file_name)
    with open(file_name, "rb") as fin:
        bytes = fin.read()  # read file as bytes
        readable_hash = hashlib.md5(bytes).hexdigest()

    return readable_hash


def load_data(filename, sep, columns):
    data = pd.read_csv(filename, sep=sep, header=None, names=columns)
    return data


def filter_data(data, user_min=None, item_min=None):
    data.dropna(how="any", inplace=True)
    if item_min is not None and item_min > 0:
        item_count = data["item"].value_counts(sort=False)
        filtered_idx = data["item"].map(lambda x: item_count[x] >= item_min)
        data = data[filtered_idx]

    if user_min is not None and user_min > 0:
        user_count = data["user"].value_counts(sort=False)
        filtered_idx = data["user"].map(lambda x: user_count[x] >= user_min)
        data = data[filtered_idx]
    return data


def remap_id(data):
    unique_user = data["user"].unique()
    user2id = pd.Series(data=range(len(unique_user)), index=unique_user)
    data["user"] = data["user"].map(user2id)

    unique_item = data["item"].unique()
    item2id = pd.Series(data=range(len(unique_item)), index=unique_item)
    data["item"] = data["item"].map(item2id)

    return data, user2id, item2id


def get_map_id(data):
    unique_user = data["user"].unique()
    user2id = pd.Series(data=range(len(unique_user)), index=unique_user)

    unique_item = data["item"].unique()
    item2id = pd.Series(data=range(len(unique_item)), index=unique_item)
    return user2id.to_dict(), item2id.to_dict()


def split_by_ratio(data, ratio=0.8, by_time=True):
    if by_time:
        data.sort_values(by=["user", "time"], inplace=True)
    else:
        data.sort_values(by=["user", "item"], inplace=True)

    first_section = []
    second_section = []
    user_grouped = data.groupby(by=["user"])
    for user, u_data in user_grouped:
        u_data_len = len(u_data)
        if not by_time:
            u_data = u_data.sample(frac=1)
        idx = math.ceil(ratio*u_data_len)
        first_section.append(u_data.iloc[:idx])
        second_section.append(u_data.iloc[idx:])

    first_section = pd.concat(first_section, ignore_index=True)
    second_section = pd.concat(second_section, ignore_index=True)

    return first_section, second_section


def split_by_loo(data, by_time=True):
    if by_time:
        data.sort_values(by=["user", "time"], inplace=True)
    else:
        data.sort_values(by=["user", "item"], inplace=True)

    first_section = []
    second_section = []
    user_grouped = data.groupby(by=["user"])
    for user, u_data in user_grouped:
        u_data_len = len(u_data)
        if u_data_len <= 3:
            first_section.append(u_data)
        else:
            if not by_time:
                u_data = u_data.sample(frac=1)
            first_section.append(u_data.iloc[:-1])
            second_section.append(u_data.iloc[-1:])

    first_section = pd.concat(first_section, ignore_index=True)
    second_section = pd.concat(second_section, ignore_index=True)

    return first_section, second_section
