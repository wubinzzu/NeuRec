def getAP(rank_list, target_items):
    hits = 0
    sum_precs = 0
    for n in range(len(rank_list)):
        if rank_list[n] in target_items:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        Min = len(rank_list) if len(rank_list)<len(target_items) else len(target_items)
        return sum_precs / Min
    else:
        return 0