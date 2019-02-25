import scipy.sparse as sp
import numpy as np
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
path="../Music"
train_matrix = None
social_matrix = None
time_matrix = None
with open(path+".rating", 'r') as f:
    for line in f.readlines():
        useridx, itemidx,rating, time= line.strip().split(",")
        
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
    print("\"num_users\": %d,\"num_items\":%d, \"num_ratings\":%d\n"%(num_users,num_items,num_ratings))
    train_matrix = sp.dok_matrix((num_users, num_items), dtype=np.int32)
    social_matrix = sp.dok_matrix((num_users, num_users), dtype=np.int32)
    time_matrix = sp.dok_matrix((num_users, num_items), dtype=np.int32)
    for u in np.arange(num_users):
        for enlement in pos_per_user[u]:
            train_matrix[u,enlement[0]]=enlement[1]
            time_matrix[u,enlement[0]] = enlement[2]
            
with open(path+".uu", 'r') as f:
    for line in f.readlines():
        useridx, socialidx = line.strip().split(",")
        if  useridx in userids and socialidx in userids:
            social_matrix[userids[useridx],userids[socialidx]]=1
            
f= open("rating.txt",'w')
f1= open("trust.txt",'w')

trainMatrix = train_matrix.tocsr()
socialMatrix = social_matrix.tocsr()
timeMatrix = time_matrix.tocsr()

for u in range(num_users):
    items_by_u = trainMatrix[u].indices
    social_by_u = socialMatrix[u].indices
    if len(items_by_u)>=10 and len(social_by_u)>=5:
        for i in items_by_u:
            rating = str(u)+","+str(i)+","+str(train_matrix[u,i])+","+str(time_matrix[u,i])+"\n"
            f.write(rating)
            
        for v in social_by_u:
            social = str(u)+","+str(v)+"\n"
            f1.write(social)

f1.close()            
f.close()           