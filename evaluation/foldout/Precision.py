def hit(ranklist, taget_items):
    count = 0
    for item in ranklist:
        if item in taget_items:
            count += 1
    return count

def getPre(ranklist,taget_items,topK):
    
    return hit(ranklist, taget_items)/topK