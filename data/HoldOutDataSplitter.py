import scipy.sparse as sp
import numpy as np
import math
from copy import deepcopy
class HoldOutDataSplitter(object):
    def __init__(self,path,separator,splitterRatio=[0.7,0.1,0.2]):
        self.path =path +".rating"
        self.separator = separator
        self.splitterRatio = splitterRatio
        if float(splitterRatio[0])+ float(splitterRatio[1]) + float(splitterRatio[2]) != 1.0:
            raise ValueError("please given a correct splitterRatio")
    def load_data_by_user_time(self):
        print("Loading interaction records from %s "%(self.path))
        pos_per_user={}
        num_ratings=0
        num_items=0
        num_users=0
        #user/item {raw id, inner id} map
        userids = {}
        itemids = {}
        # inverse views of userIds, itemIds, 
        idusers = {}
        iditems={}
        with open(self.path, 'r') as f:
            for line in f.readlines():
                useridx, itemidx,rating, time= line.strip().split(self.separator)
                
                num_ratings+=1
                if  itemidx not in itemids:
                    iditems[num_items]=itemidx
                    itemids[itemidx] = num_items
                    num_items+=1

                if useridx not in userids:
                    idusers[num_users]=useridx
                    userids[useridx]=num_users
                    num_users+=1
                    pos_per_user[userids[useridx]]=[]
                pos_per_user[userids[useridx]].append([itemids[itemidx],rating,int(time)])
                # rating_matrix[self.userids[useridx],self.itemids[itemidx]] = rating
            print("Sorting interactions for each users")

        for u in range(num_users):
            pos_per_user[u]=sorted(pos_per_user[u], key=lambda d: d[2])
        print("\"num_users\": %d,\"num_items\":%d, \"num_ratings\":%d\n"%(num_users,num_items,num_ratings))
        userseq = deepcopy(pos_per_user)
        train_dict = {}
        valid_dict = {}
        train_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        valid_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        test_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        time_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        for u in range(num_users):
            num_ratings_by_user = len(pos_per_user[u])
            num_valid_ratings = math.floor(float(self.splitterRatio[1])*num_ratings_by_user)
            num_test_ratings = math.floor(float(self.splitterRatio[2])*num_ratings_by_user)
            if len(pos_per_user[u]) > 3 and  num_valid_ratings>1 and num_test_ratings >1:
                for _ in range(num_test_ratings):
                    test_item=pos_per_user[u][-1]
                    pos_per_user[u].pop() 
                    test_matrix[u,test_item[0]] = test_item[1] 
                    time_matrix[u,test_item[0]] = test_item[2]
                    
                items = []    
                for _ in range(num_valid_ratings):
                    valid_item=pos_per_user[u][-1]
                    pos_per_user[u].pop() 
                    valid_matrix[u,valid_item[0]] = valid_item[1]
                    time_matrix[u,valid_item[0]] = valid_item[2] 
                    items.append(valid_item[0]) 
                items.reverse()
                valid_dict[u] =  items
            items = []
            for enlement in pos_per_user[u]:
                items.append(enlement[0])
                train_matrix[u,enlement[0]]=enlement[1]
                time_matrix[u,enlement[0]] = enlement[2]
            train_dict[u]=items
        return train_matrix,train_dict,valid_dict,valid_matrix,test_matrix,userseq,userids,itemids,time_matrix