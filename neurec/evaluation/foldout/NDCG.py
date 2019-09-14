import math
def getNDCG(rank_list, target_items):
    dcg = 0
    idcg = IDCG(len(target_items))
    for i in range(len(rank_list)):
        item_id= rank_list[i]
        if (item_id not in target_items):
            continue
        rank = i +1
        dcg +=1/math.log(rank+1,2)
    return dcg/idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg +=1/math.log(i+2,2)
    return idcg