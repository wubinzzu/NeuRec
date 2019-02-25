import scipy.sparse as sp
import numpy as np
from copy import deepcopy
class LeaveOneOutDataSplitter(object):
    def __init__(self,path,separator):
        self.path =path
        self.separator = separator
    def load_training_file_as_matrix(self):
            '''
            Read .rating file and Return dok matrix.
            The first line of .rating file is: num_users\t num_items
            '''
            # Get number of users and items
            num_users, num_items = 0, 0
            with open(self.path+".train.rating", "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.split(self.separator)
                    u, i = int(arr[0]), int(arr[1])
                    num_users = max(num_users, u)
                    num_items = max(num_items, i)
                    line = f.readline()
            # Construct matrix
            mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
            with open(self.path+".train.rating", "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.split("\t")
                    user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                    line = f.readline()
            print ("already load the trainMatrix...")
            return mat

    def load_training_file_as_list(self):
            # Get number of users and items
            u_ = 0
            lists, items = [], []
            with open(self.path+".train.rating", "r") as f:
                line = f.readline()
                index = 0
                while line != None and line != "":
                    arr = line.split(self.separator)
                    u, i = int(arr[0]), int(arr[1])
                    if u_ < u:
                        index = 0
                        lists.append(items)
                        items = []
                        u_ += 1
                    index += 1
                    items.append(i)
                    line = f.readline()
            lists.append(items)
            print ("already load the trainList...")
            return lists

    def load_testrating_file_as_list(self):
        ratingList = []
        with open(self.path+".test.rating", "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(self.separator)
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_validrating_file_as_list(self):
        ratingList = []
        with open(self.path+".valid.rating", "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(self.separator)
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    def load_negative_file(self):
        negativeList = []
        with open(self.path+".negative", "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(self.separator)
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    def load_data_by_user_time(self):
        print("Loading interaction records from %s "%(self.path))
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
                pos_per_user[userids[useridx]].append((itemids[itemidx],rating,int(time)))
                # rating_matrix[self.userids[useridx],self.itemids[itemidx]] = rating
            print("Sorting interactions for each users")

            for u in np.arange(num_users):
                pos_per_user[u]=sorted(pos_per_user[u], key=lambda d: d[2])
            print("\"num_users\": %d,\"num_items\":%d, \"num_ratings\":%d\n"%(num_users,num_items,num_ratings))
            userseq = deepcopy(pos_per_user)
            train_dict = {}
            valid_dict = {}
            train_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
            valid_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
            test_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
            time_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
            for u in np.arange(num_users):
                if len(pos_per_user[u])<3:
                    test_item=-1
                    valid_item=-1
                    continue
                
                test_item=pos_per_user[u][-1]
                pos_per_user[u].pop()
                
                valid_item=pos_per_user[u][-1]
                pos_per_user[u].pop()

                
                valid_matrix[u,valid_item[0]] = valid_item[1]
                valid_dict[u] = valid_item[0]
                time_matrix[u,valid_item[0]] = valid_item[2]
                test_matrix[u,test_item[0]] = test_item[1]
                time_matrix[u,test_item[0]] = test_item[2]
                items = []
                for enlement in pos_per_user[u]:
                    items.append(enlement[0])
                    train_matrix[u,enlement[0]]=enlement[1]
                    time_matrix[u,enlement[0]] = enlement[2]
                train_dict[u]=items  
        return train_matrix,train_dict,valid_dict,valid_matrix,test_matrix,userseq,userids,itemids,time_matrix