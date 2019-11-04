# def hit(gt_items, pred_items):
#     count = 0
#     for item in pred_items:
#         if item in gt_items:
#             count += 1
#     return count

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0