import numpy as np
from neurec.util.tool import randint_choice
def _get_pairwise_all_data(dataset):
    user_input, item_input_pos, item_input_neg = [], [], []
    train_matrix = dataset.train_matrix
    num_items = dataset.num_items
    num_users = dataset.num_users
    
    for u in range(num_users):
        items_by_u = train_matrix[u].indices
        num_items_by_u = len(items_by_u)
        if num_items_by_u > 0:
            user_input.extend([u]*num_items_by_u)
            item_input_pos.extend(items_by_u)
            item_input_neg.extend(randint_choice(num_items, num_items_by_u, replace=True, exclusion = items_by_u))
            
    return user_input, item_input_pos, item_input_neg 


def _get_pairwise_all_highorder_data(dataset, high_order, train_dict):
    user_input, item_input_pos, item_input_recents, item_input_neg = [], [], [], []
    num_items = dataset.num_items
    num_users = dataset.num_users
    for u in range(num_users):
        items_by_u = train_dict[u]
        num_items_by_u = len(items_by_u)
        
        if num_items_by_u > high_order: 
            user_input.extend([u]*(num_items_by_u-high_order)) 
            item_input_pos.extend(items_by_u[high_order:])
            item_input_neg.extend(randint_choice(num_items, (num_items_by_u-high_order), replace=True, exclusion = items_by_u))
            
            for idx in range(high_order, num_items_by_u):
                item_input_recents.append(items_by_u[idx-high_order:idx])
                
    return user_input, item_input_pos, item_input_recents, item_input_neg 


def _get_pairwise_all_firstorder_data(dataset, train_dict):
    user_input, item_input_pos, item_input_recent, item_input_neg = [], [], [], []
    num_items = dataset.num_items
    num_users = dataset.num_users
    for u in range(num_users):
        items_by_u = train_dict[u]
        num_items_by_u = len(items_by_u)
        if num_items_by_u > 1: 
            user_input.extend([u]*(num_items_by_u-1)) 
            item_input_pos.extend(items_by_u[1:])
            item_input_recent.extend(items_by_u[:-1])
            item_input_neg.extend(randint_choice(num_items, (num_items_by_u-1), replace=True, exclusion = items_by_u))
    return user_input, item_input_pos, item_input_recent, item_input_neg


def _get_pointwise_all_data(dataset, num_negatives):
    user_input, item_input, labels = [], [], []
    train_matrix = dataset.train_matrix
    num_items = dataset.num_items
    num_users = dataset.num_users
    
    for u in range(num_users):
        items_by_u = train_matrix[u].indices
        num_items_by_u = len(items_by_u)
        if num_items_by_u > 0:
            negative_items = randint_choice(num_items, num_items_by_u*num_negatives, replace=True, exclusion = items_by_u)
            index = 0
            for i in items_by_u:
                # positive instance
                user_input.append(u)
                item_input.append(i)
                labels.append(1)
                # negative instance
                user_input.extend([u]*num_negatives)
                item_input.extend(negative_items[index:index+num_negatives])
                labels.extend([0]*num_negatives)
                index = index +1
    return user_input, item_input, labels

def _get_pointwise_all_highorder_data(dataset, high_order, num_negatives, train_dict):
    user_input, item_input, item_input_recents, labels = [], [], [], []
    num_items = dataset.num_items
    num_users = dataset.num_users
    
    for u in range(num_users):
        items_by_u = train_dict[u]
        num_items_by_u = len(items_by_u)
        if num_items_by_u > high_order:
            
            negative_items = randint_choice(num_items, (num_items_by_u-high_order)*num_negatives, replace=True, exclusion = items_by_u)
            index = 0   
            for idx in range(high_order, num_items_by_u):
                user_input.append(u)
                i = items_by_u[idx] # item id 
                item_input.append(i)
                item_input_recents.append(items_by_u[idx-high_order:idx])
                labels.append(1)
                user_input.extend([u]*num_negatives)
                item_input.extend(negative_items[index:index+num_negatives])
                item_input_recents.extend([items_by_u[idx-high_order:idx]]*num_negatives)
                labels.extend([0]*num_negatives)
                index = index +1
                
    return user_input, item_input, item_input_recents, labels 

def _get_pointwise_all_firstorder_data(dataset, num_negatives, train_dict):
    user_input,item_input,item_input_recent,labels = [],[],[],[]
    num_items = dataset.num_items
    num_users = dataset.num_users
    for u in range(num_users):
        items_by_user = train_dict[u]
        num_items_by_u = len(items_by_user)
        negative_items = randint_choice(num_items, (num_items_by_u-1)*num_negatives, replace=True, exclusion = items_by_user)
        index = 0
        for idx in range(1,num_items_by_u):
            i = items_by_user[idx] # item id 
            user_input.append(u)
            item_input.append(i)
            item_input_recent.append(items_by_user[idx-1])
            labels.append(1)
            # negative instance
            user_input.extend([u]*num_negatives)
            item_input.extend(negative_items[index:index+num_negatives])
            item_input_recent.extend([items_by_user[idx-1]]*num_negatives)
            labels.extend([0]*num_negatives)
            index = index +1
    user_input = np.array(user_input, dtype=np.int32)
    item_input = np.array(item_input, dtype=np.int32)
    item_input_recent = np.array(item_input_recent, dtype=np.int32)
    labels = np.array(labels, dtype=np.float32)
    num_training_instances = len(user_input)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input = user_input[shuffle_index]
    item_input = item_input[shuffle_index]
    item_input_recent = item_input_recent[shuffle_index]
    labels = labels[shuffle_index]
    return user_input,item_input,item_input_recent,labels

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
    
    user_input_pos = np.array(user_input_pos)
    user_input_neg = np.array(user_input_neg)
    num_idx_pos = np.array(num_idx_pos,dtype=np.int32)
    num_idx_neg = np.array(num_idx_neg,dtype=np.int32)
    item_input_pos = np.array(item_input_pos, dtype=np.int32)
    item_input_neg = np.array(item_input_neg, dtype=np.int32)
    num_training_instances = len(user_input_pos)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input_pos = user_input_pos[shuffle_index]
    user_input_neg = user_input_neg[shuffle_index]
    num_idx_pos = num_idx_pos[shuffle_index]
    num_idx_neg = num_idx_neg[shuffle_index]
    item_input_pos = item_input_pos[shuffle_index]
    item_input_neg = item_input_neg[shuffle_index]    
                
    return user_input_pos, user_input_neg, num_idx_pos, num_idx_neg, item_input_pos, item_input_neg

def _get_pointwise_all_likefism_data(dataset, num_negatives, train_dict):
    user_input,num_idx,item_input,labels = [],[],[],[]
    num_users = dataset.num_users
    num_items = dataset.num_items
    for u in range(num_users):
        items_by_user = train_dict[u].copy()
        num_items_by_u = len(items_by_user)  
        negative_items = randint_choice(num_items, num_items_by_u*num_negatives, replace=True, exclusion = items_by_user) 
        index = 0
        for idx in range(num_items_by_u):
            # negative instances
            user_input.extend([items_by_user]*num_negatives)
            item_input.extend(negative_items[index:index+num_negatives])
            num_idx.extend([num_items_by_u]*num_negatives)
            labels.extend([0]*num_negatives)
            
            i = items_by_user[idx]
            items_by_user.remove(i)
            user_input.append(items_by_user)
            item_input.append(i)
            num_idx.append(num_items_by_u-1)
            labels.append(1)
            items_by_user.append(i)
    user_input = np.array(user_input)
    num_idx = np.array(num_idx, dtype=np.int32)
    item_input = np.array(item_input, dtype=np.int32)
    labels = np.array(labels, dtype=np.float32)
    num_training_instances = len(user_input)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input = user_input[shuffle_index]
    num_idx = num_idx[shuffle_index]
    item_input = item_input[shuffle_index]
    labels = labels[shuffle_index]
    return user_input,num_idx,item_input,labels
 

def _get_pointwise_batch_likefism_data(user_input, num_items, num_idx, item_input, labels, num_batch, batch_size):
    num_training_instances = len(user_input)
    id_start = num_batch * batch_size
    id_end = (num_batch + 1) * batch_size
    if id_end>num_training_instances:
        id_end=num_training_instances
    bat_users = user_input[id_start:id_end].tolist()
    bat_idx = num_idx[id_start:id_end]
    max_by_user = max(bat_idx)
    for i in range(len(bat_users)):
        bat_users[i] = bat_users[i] + [num_items] * (max_by_user - len(bat_users[i]))
    bat_items = item_input[id_start:id_end]
    bat_labels = labels[id_start:id_end]
    return bat_users, bat_idx, bat_items, bat_labels


def _get_pairwise_batch_likefism_data(user_input_pos,user_input_neg,num_items, num_idx_pos, num_idx_neg, item_input_pos,item_input_neg,num_batch,batch_size):
    num_training_instances = len(user_input_pos)
    id_start = num_batch * batch_size
    id_end = (num_batch + 1) * batch_size
    if id_end>num_training_instances:
        id_end=num_training_instances
    bat_idx_pos = num_idx_pos[id_start:id_end]
    bat_idx_neg = num_idx_neg[id_start:id_end]
    max_pos = max(bat_idx_pos)  
    max_neg = max(bat_idx_neg) 
    bat_users_pos = user_input_pos[id_start:id_end].tolist()
    bat_users_neg = user_input_neg[id_start:id_end].tolist()
    for i in range(len(bat_users_pos)):
        bat_users_pos[i] = bat_users_pos[i] + \
        [num_items] * (max_pos - len(bat_users_pos[i]))
        
    for i in range(len(bat_users_neg)):
        bat_users_neg[i] = bat_users_neg[i] + \
        [num_items] * (max_neg - len(bat_users_neg[i]))  
    
    bat_items_pos = item_input_pos[id_start:id_end]
    bat_items_neg = item_input_neg[id_start:id_end]
    return bat_users_pos,bat_users_neg,bat_idx_pos,bat_idx_neg,bat_items_pos,bat_items_neg