import numpy as np
def _get_pairwise_all_data(dataset):
    user_input, item_input_pos,item_input_neg = [], [], []
    trainMatrix = dataset.trainMatrix
    num_items = dataset.num_items
    for (u, i) in trainMatrix.keys():
        user_input.append(u)
        item_input_pos.append(i)
        j = np.random.randint(num_items)
        while (u, j) in trainMatrix.keys():
            j = np.random.randint(num_items)
        item_input_neg.append(j)
    user_input = np.array(user_input, dtype=np.int32)
    item_input_pos = np.array(item_input_pos, dtype=np.int32)
    item_input_neg = np.array(item_input_neg, dtype=np.int32)
    num_training_instances = len(user_input)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input=user_input[shuffle_index]
    item_input_pos=item_input_pos[shuffle_index]
    item_input_neg=item_input_neg[shuffle_index]
    return user_input, item_input_pos,item_input_neg 

def _get_pairwise_all_highorder_data(dataset,high_order):
    user_input, item_input_pos,item_input_recents,item_input_neg = [], [], [],[]
    trainMatrix = dataset.trainMatrix
    num_items = dataset.num_items
    num_users = dataset.num_users
    trainDict = dataset.trainDict
    for u in range(num_users):
        items_by_user = trainDict[u]
        for idx in range(high_order,len(items_by_user)):
            i = items_by_user[idx] # item id 
            # positive instance
            user_input.append(u)
            item_input_pos.append(i)
            item_input_recent = []
            for t in range(1,high_order+1):
                item_input_recent.append(items_by_user[idx-t])
            item_input_recents.append(item_input_recent)
            j = np.random.randint(num_items)
            while (u, j) in trainMatrix.keys():
                j = np.random.randint(num_items)
            item_input_neg.append(j)
    user_input = np.array(user_input, dtype=np.int32)
    item_input_pos = np.array(item_input_pos, dtype=np.int32)
    item_input_recents = np.array(item_input_recents)
    item_input_neg = np.array(item_input_neg, dtype=np.int32)
    num_training_instances = len(user_input)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input = user_input[shuffle_index]
    item_input_pos = item_input_pos[shuffle_index]
    item_input_recents = item_input_recents[shuffle_index]
    item_input_neg = item_input_neg[shuffle_index]    
    return user_input, item_input_pos,item_input_recents,item_input_neg 

def _get_pairwise_all_firstorder_data(dataset):
    user_input, item_input_pos,item_input_recent,item_input_neg = [], [], [],[]
    trainMatrix = dataset.trainMatrix
    num_items = dataset.num_items
    num_users = dataset.num_users
    train_dict = dataset.trainDict
    for u in range(num_users):
        items_by_user = train_dict[u]
        for idx in range(1,len(items_by_user)):
            i = items_by_user[idx] # item id 
            # positive instance
            user_input.append(u)
            item_input_pos.append(i)
            item_input_recent.append(items_by_user[idx-1])
            j = np.random.randint(num_items)
            while (u, j) in trainMatrix.keys():
                j = np.random.randint(num_items)
            item_input_neg.append(j)
    user_input = np.array(user_input, dtype=np.int32)
    item_input_pos = np.array(item_input_pos, dtype=np.int32)
    item_input_recent = np.array(item_input_recent,dtype=np.int32)
    item_input_neg = np.array(item_input_neg, dtype=np.int32)
    num_training_instances = len(user_input)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input = user_input[shuffle_index]
    item_input_pos = item_input_pos[shuffle_index]
    item_input_recent = item_input_recent[shuffle_index]
    item_input_neg = item_input_neg[shuffle_index]    
    return user_input,item_input_pos,item_input_recent,item_input_neg 

def _get_pairwise_all_likefism_data(dataset):
    user_input_pos,user_input_neg, num_idx_pos, num_idx_neg, item_input_pos,item_input_neg = [], [], [],[],[],[]
    trainMatrix = dataset.trainMatrix
    num_items = dataset.num_items
    num_users = dataset.num_users
    trainDict = dataset.trainDict
    for u in range(num_users):
        items_by_user = trainDict[u].copy()
        size = len(items_by_user)   
        for i in items_by_user:
            j = np.random.randint(num_items)
            while (u,j) in trainMatrix.keys():
                j = np.random.randint(num_items)
            user_input_neg.append(items_by_user)
            num_idx_neg.append(size)
            item_input_neg.append(j)
            
            items_by_user.remove(i)
            user_input_pos.append(items_by_user)
            num_idx_pos.append(size-1)
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
    return user_input_pos,user_input_neg, num_idx_pos, num_idx_neg, item_input_pos,item_input_neg

def _get_pointwise_all_likefism_data(dataset,num_negatives):
    user_input,num_idx,item_input,lables = [],[],[],[]
    num_users = dataset.num_users
    num_items = dataset.num_items
    trainMatrix = dataset.trainMatrix
    trainDict = dataset.trainDict
    for u in range(num_users):
        items_by_user = trainDict[u].copy()
        size = len(items_by_user)   
        for i in items_by_user:
            # negative instances
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u,j) in trainMatrix.keys():
                    j = np.random.randint(num_items)
                user_input.append(items_by_user)
                item_input.append(j)
                num_idx.append(size)
                lables.append(0)
            items_by_user.remove(i)
            user_input.append(items_by_user)
            item_input.append(i)
            num_idx.append(size-1)
            lables.append(1)
    user_input = np.array(user_input)
    num_idx = np.array(num_idx, dtype=np.int32)
    item_input = np.array(item_input, dtype=np.int32)
    lables = np.array(lables, dtype=np.float32)
    num_training_instances = len(user_input)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input = user_input[shuffle_index]
    num_idx = num_idx[shuffle_index]
    item_input = item_input[shuffle_index]
    lables = lables[shuffle_index]
    return user_input,num_idx,item_input,lables

def _get_pointwise_all_data(dataset,num_negatives):
    user_input,item_input,lables = [],[],[]
    trainMatrix = dataset.trainMatrix
    num_items= dataset.num_items
    for (u, i) in trainMatrix.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        lables.append(1)
        # negative instance
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in trainMatrix.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            lables.append(0)
    user_input = np.array(user_input, dtype=np.int32)
    item_input = np.array(item_input, dtype=np.int32)
    lables = np.array(lables, dtype=np.float32)
    num_training_instances = len(user_input)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input=user_input[shuffle_index]
    item_input=item_input[shuffle_index]
    lables = lables[shuffle_index]
    return user_input,item_input,lables

def _get_pointwise_all_highorder_data(dataset,high_order,num_negatives):
    user_input, item_input,item_input_recents,lables = [], [], [], []
    trainMatrix = dataset.trainMatrix
    num_items = dataset.num_items
    num_users = dataset.num_users
    trainDict = dataset.trainDict
    for u in range(num_users):
        items_by_user = trainDict[u]
        for idx in range(high_order,len(items_by_user)):
            i = items_by_user[idx] # item id 
            # positive instance
            user_input.append(u)
            item_input.append(i)
            item_input_recent = []
            for t in range(1,high_order+1):
                item_input_recent.append(items_by_user[idx-t])
            item_input_recents.append(item_input_recent)
            lables.append(1)
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in trainMatrix.keys():
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                item_input_recent = []
                for t in range(1,high_order+1):
                    item_input_recent.append(items_by_user[idx-t])
                item_input_recents.append(item_input_recent)
                lables.append(0)        
    user_input = np.array(user_input, dtype=np.int32)
    item_input = np.array(item_input, dtype=np.int32)
    item_input_recents = np.array(item_input_recents)
    lables = np.array(lables, dtype=np.int32)
    num_training_instances = len(user_input)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input = user_input[shuffle_index]
    item_input = item_input[shuffle_index]
    item_input_recents = item_input_recents[shuffle_index]
    lables = lables[shuffle_index]    
    return user_input, item_input, item_input_recents, lables 

def _get_pointwise_all_firstorder_data(dataset,num_negatives):
    trainMatrix = dataset.trainMatrix
    trainDict = dataset.trainDict 
    user_input,item_input,item_input_recent,lables = [],[],[],[]
    num_items = dataset.num_items
    num_users = dataset.num_users
    for u in range(num_users):
        items_by_user = trainDict[u]
        for idx in range(1,len(items_by_user)):
            i = items_by_user[idx] # item id 
            user_input.append(u)
            item_input.append(i)
            item_input_recent.append(items_by_user[idx-1])
            lables.append(1)
            # negative instance
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in trainMatrix.keys():
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                item_input_recent.append(items_by_user[idx-1])
                lables.append(0)
    user_input = np.array(user_input, dtype=np.int32)
    item_input = np.array(item_input, dtype=np.int32)
    item_input_recent = np.array(item_input_recent, dtype=np.int32)
    lables = np.array(lables, dtype=np.float32)
    num_training_instances = len(user_input)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input = user_input[shuffle_index]
    item_input = item_input[shuffle_index]
    item_input_recent = item_input_recent[shuffle_index]
    lables = lables[shuffle_index]
    return user_input,item_input,item_input_recent,lables
   
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


def _get_pairwise_batch_seqdata(user_input,item_input_pos,item_input_recent,item_input_neg,num_batch,batch_size):
    num_training_instances = len(user_input)
    id_start = num_batch * batch_size
    id_end = (num_batch + 1) * batch_size
    if id_end>num_training_instances:
        id_end=num_training_instances
    bat_users = user_input[id_start:id_end]
    bat_items_pos = item_input_pos[id_start:id_end]
    bat_item_recent = item_input_recent[id_start:id_end]
    bat_items_neg = item_input_neg[id_start:id_end]
    return bat_users,bat_items_pos,bat_item_recent,bat_items_neg

 
def _get_pointwise_batch_seqdata(user_input,item_input,item_input_recent,lables,num_batch,batch_size):
    num_training_instances = len(user_input)
    id_start = num_batch * batch_size
    id_end = (num_batch + 1) * batch_size
    if id_end>num_training_instances:
        id_end=num_training_instances
    bat_users = user_input[id_start:id_end]
    bat_items = item_input[id_start:id_end]
    bat_item_recent = item_input_recent[id_start:id_end]
    bat_lables = lables[id_start:id_end]
    return bat_users,bat_items,bat_item_recent,bat_lables
 
def _get_pointwise_batch_likefism_data(user_input,num_items,num_idx,item_input,lables,num_batch,batch_size):
    num_training_instances =len(user_input) 
    id_start = num_batch * batch_size
    id_end = (num_batch + 1) * batch_size
    if id_end>num_training_instances:
        id_end=num_training_instances
    bat_users = user_input[id_start:id_end].tolist()
    bat_idx = num_idx[id_start:id_end]
    max_by_user = max(bat_idx)
    for i in range(len(bat_users)):
        bat_users[i] = bat_users[i] + \
        [num_items] * (max_by_user - len(bat_users[i]))
    bat_items = item_input[id_start:id_end]
    bat_lables = lables[id_start:id_end]
    return bat_users,bat_idx,bat_items,bat_lables
   
def _get_pointwise_batch_data(user_input,item_input,lables,num_batch,batch_size):
    num_training_instances =len(user_input) 
    id_start = num_batch * batch_size
    id_end = (num_batch + 1) * batch_size
    if id_end>num_training_instances:
        id_end=num_training_instances
    bat_users = user_input[id_start:id_end]
    bat_items = item_input[id_start:id_end]
    bat_lables = lables[id_start:id_end]
    return bat_users, bat_items,bat_lables

def _get_pairwise_batch_data(user_input,item_input_pos, item_input_neg, num_batch, batch_size):
    num_training_instances =len(user_input) 
    id_start = num_batch * batch_size
    id_end = (num_batch + 1) * batch_size
    if id_end>num_training_instances:
        id_end=num_training_instances
    bat_users = user_input[id_start:id_end]
    bat_items_pos = item_input_pos[id_start:id_end]
    bat_items_neg = item_input_neg[id_start:id_end]
    return bat_users, bat_items_pos,bat_items_neg