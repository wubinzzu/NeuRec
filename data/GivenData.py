import scipy.sparse as sp
import numpy as np
class GivenData(object):
    def __init__(self,path,separator):
        self.path =path +".rating"
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