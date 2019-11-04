def hit(ranklist,taget_items):
    count = 0
    for item in ranklist:
        if item in taget_items:
            count += 1
    return count

def getRec(ranklist,taget_items):
    
    return hit(ranklist, taget_items)/len(taget_items)