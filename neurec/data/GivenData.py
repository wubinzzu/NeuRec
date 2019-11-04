import scipy.sparse as sp
import numpy as np
from neurec.util import reader
import logging

class GivenData(object):
    def __init__(self, path, dataset_name, separator,threshold):
        self.path =path
        self.dataset_name = dataset_name
        self.separator = separator
        self.threshold = threshold
        global num_items,num_users,userids,itemids,idusers,iditems

    def load_pre_splitter_data(self):
        pos_per_user={}
        num_items,num_users = 0,0
        userids,itemids,idusers,iditems = {},{},{},{}
        # Get number of users and items

        data = reader.lines(self.path + '/' + self.dataset_name)

        for line in data:
            useridx, itemidx, rating, time= line.strip().split(self.separator)
            if float(rating)>= self.threshold:
                if  itemidx not in itemids:
                    iditems[num_items]=itemidx
                    itemids[itemidx] = num_items
                    num_items+=1

                if useridx not in userids:
                    idusers[num_users]=useridx
                    userids[useridx]=num_users
                    num_users+=1
                    pos_per_user[userids[useridx]]=[]
                pos_per_user[userids[useridx]].append([itemids[itemidx],1,int(time)])
            else:
                if  itemidx not in itemids:
                    iditems[num_items]=itemidx
                    itemids[itemidx] = num_items
                    num_items+=1

                if useridx not in userids:
                    idusers[num_users]=useridx
                    userids[useridx]=num_users
                    num_users+=1
                    pos_per_user[userids[useridx]]=[]
                pos_per_user[userids[useridx]].append((itemids[itemidx],rating,int(time)))

            train_dict = {}
            for u in range(num_users):
                pos_per_user[u] =sorted(pos_per_user[u], key=lambda d: d[2])
                items = []
                for enlement in pos_per_user[u]:
                    items.append(enlement[0])
                train_dict[u] = items

        with open(self.path+".test.rating", 'r') as f:
            for line in f.readlines():
                useridx, itemidx,rating, time= line.strip().split(self.separator)
                if float(rating)>= self.threshold:
                    if  itemidx not in itemids:
                        iditems[num_items]=itemidx
                        itemids[itemidx] = num_items
                        num_items+=1

                    if useridx not in userids:
                        idusers[num_users]=useridx
                        userids[useridx]=num_users
                        num_users+=1
                        pos_per_user[userids[useridx]]=[]
                    pos_per_user[userids[useridx]].append([itemids[itemidx],1,int(time)])
                else :
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
        for u in range(num_users):
            pos_per_user[u]=sorted(pos_per_user[u], key=lambda d: d[2])

        train_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        time_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        with open(self.path+".train.rating", "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating,time = userids[arr[0]], itemids[arr[1]], float(arr[2]), float(arr[3])
                if float(rating)>= self.threshold:
                    train_matrix[user, item] = 1

                else :
                    train_matrix[user, item] = rating
                time_matrix[user, item] = time
                line = f.readline()
        logging.info("already load the trainMatrix...")

        test_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        with open(self.path+".test.rating", "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating, time = userids[arr[0]], itemids[arr[1]], float(arr[2]), float(arr[3])
                if float(rating)>= self.threshold:
                    test_matrix[user, item] = 1
                else :
                    test_matrix[user, item] = rating
                time_matrix[user, item] = time
                line = f.readline()
        logging.info("already load the trainMatrix...")

        return train_matrix,train_dict,test_matrix,pos_per_user,userids,itemids,time_matrix
