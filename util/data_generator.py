import numpy as np
from util.tool import randint_choice


def _get_pairwise_all_likefism_data(dataset):
    user_input_pos, user_input_neg, num_idx_pos, num_idx_neg, item_input_pos, item_input_neg = [], [], [], [], [], []
    num_items = dataset.num_items
    num_users = dataset.num_users
    train_matrix = dataset.train_matrix
    for u in range(num_users):
        items_by_u = train_matrix[u].indices.copy().tolist()
        num_items_by_u = len(items_by_u)
        if num_items_by_u > 1: 
            negative_items = randint_choice(num_items, num_items_by_u, replace=True, exclusion = items_by_u)
        
            for index, i in enumerate(items_by_u):
                j = negative_items[index]
                user_input_neg.append(items_by_u)
                num_idx_neg.append(num_items_by_u)
                item_input_neg.append(j)
                
                items_by_u.remove(i)
                user_input_pos.append(items_by_u)
                num_idx_pos.append(num_items_by_u-1)
                item_input_pos.append(i)  
                
    return user_input_pos, user_input_neg, num_idx_pos, num_idx_neg, item_input_pos, item_input_neg

def _get_pointwise_all_likefism_data(dataset, num_negatives, train_dict):
    user_input,num_idx,item_input,labels = [],[],[],[]
    num_users = dataset.num_users
    num_items = dataset.num_items
    for u in range(num_users):
        items_by_user = train_dict[u].copy()
        items_set = set(items_by_user)
        size = len(items_by_user)   
        for i in items_by_user:
            # negative instances
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while j in items_set:
                    j = np.random.randint(num_items)
                user_input.append(items_by_user)
                item_input.append(j)
                num_idx.append(size)
                labels.append(0)
            items_by_user.remove(i)
            user_input.append(items_by_user)
            item_input.append(i)
            num_idx.append(size-1)
            labels.append(1)
    return user_input,num_idx,item_input,labels

def _get_pairwise_all_likefossil_data(dataset, high_order, train_dict):
    user_input_id,user_input_pos,user_input_neg, num_idx_pos, num_idx_neg, item_input_pos,item_input_neg,item_input_recents = [],[], [], [],[],[],[],[]
    for u in range(dataset.num_users):
        items_by_user = train_dict[u].copy()
        num_items_by_u = len(items_by_user)
        if  num_items_by_u > high_order: 
            negative_items = randint_choice(dataset.num_items, num_items_by_u, replace=True, exclusion = items_by_user)
            for idx in range(high_order,len(train_dict[u])):
                i = train_dict[u][idx] # item id 
                item_input_recent = []
                for t in range(1,high_order+1):
                    item_input_recent.append(train_dict[u][idx-t])
                item_input_recents.append(item_input_recent)
                j = negative_items[idx]
                user_input_neg.append(items_by_user)
                num_idx_neg.append(num_items_by_u)
                item_input_neg.append(j)
                
                items_by_user.remove(i)
                user_input_id.append(u)
                user_input_pos.append(items_by_user)
                num_idx_pos.append(num_items_by_u-1)
                item_input_pos.append(i)
                
    return user_input_id,user_input_pos,user_input_neg, num_idx_pos, num_idx_neg, item_input_pos,item_input_neg,item_input_recents

def _get_pointwise_all_likefossil_data(dataset, high_order, num_negatives, train_dict):
    user_input_id,user_input,num_idx,item_input,item_input_recents,labels = [],[],[],[],[],[]
    for u in range(dataset.num_users):
        items_by_user = train_dict[u].copy()
        items_set = set(items_by_user)
        size = len(items_by_user)   
        for idx in range(high_order,len(train_dict[u])):
            i = train_dict[u][idx] # item id 
            item_input_recent = []
            for t in range(1,high_order+1):
                item_input_recent.append(train_dict[u][idx-t])
            # negative instances
            for _ in range(num_negatives):
                j = np.random.randint(dataset.num_items)
                while j in items_set:
                    j = np.random.randint(dataset.num_items)
                user_input_id.append(u)
                user_input.append(items_by_user)
                item_input_recents.append(item_input_recent)
                item_input.append(j)
                num_idx.append(size)
                labels.append(0)
            items_by_user.remove(i)
            user_input.append(items_by_user)
            user_input_id.append(u)
            item_input_recents.append(item_input_recent)
            item_input.append(i)
            num_idx.append(size-1)
            labels.append(1)
    return user_input_id,user_input,num_idx,item_input,item_input_recents,labels