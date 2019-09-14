def getMRR(rank_list, target_items):
    for i in range(len(rank_list)):
        item_idx = rank_list[i] 
        if item_idx in target_items:
            return 1/(i+1)
        else:
            return 0