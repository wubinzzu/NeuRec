import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from util.Logger import logger
class LeaveOneOutDataSplitter(object):
    def __init__(self,path,data_format,separator, threshold):
        self.path =path
        self.data_format = data_format
        self.separator = separator
        self.threshold = threshold
    def load_data_by_user_time(self):
        logger.info("Loading interaction records from %s "%(self.path))
        pos_per_user = {}
        num_ratings=0
        num_items=0
        num_users=0
        #user/item {raw id, inner id} map
        userids = {}
        itemids = {}
        # inverse views of userIds, itemIds, 
        idusers = {}
        iditems={}
        with open(self.path+".rating", 'r') as f:
            for line in f.readlines():
                if self.data_format == "UIRT":
                    useridx, itemidx,rating,time= line.strip().split(self.separator)
                    if float(rating) < self.threshold:
                        continue
                elif self.data_format == "UIT":
                    useridx, itemidx,time= line.strip().split(self.separator)
                    rating = 1
                elif self.data_format == "UIR":
                    useridx, itemidx,rating = line.strip().split(self.separator)
                    if float(rating) < self.threshold:
                        continue
                elif self.data_format == "UI":
                    useridx, itemidx = line.strip().split(self.separator)
                    rating = 1
                    
                else:
                    print("please choose a correct data format. ")
                   
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
                if  self.data_format == "UIRT" or self.data_format == "UIT":
                    pos_per_user[userids[useridx]].append((itemids[itemidx],rating,int(float(time))))
                    
                else:
                    pos_per_user[userids[useridx]].append((itemids[itemidx],rating,1))
                    
        if  self.data_format == "UIRT" or self.data_format == "UIT":
            for u in np.arange(num_users):
                pos_per_user[u]=sorted(pos_per_user[u], key=lambda d: d[2])
        logger.info("\"num_users\": %d,\"num_items\":%d, \"num_ratings\":%d\n"%(num_users,num_items,num_ratings))
        userseq = deepcopy(pos_per_user)
        train_dict = {}
        time_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        train_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        test_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        for u in np.arange(num_users):
            if len(pos_per_user[u])<2:
                test_item=-1
                continue
            test_item=pos_per_user[u][-1]
            pos_per_user[u].pop()
            
            test_matrix[u,test_item[0]] = test_item[1]
            time_matrix[u,test_item[0]] = test_item[2]
            items = []
            for enlement in pos_per_user[u]:
                items.append(enlement[0])
                train_matrix[u,enlement[0]]=enlement[1]
                time_matrix[u,enlement[0]] = enlement[2]
        return train_matrix,train_dict,test_matrix,userseq,userids,itemids,time_matrix