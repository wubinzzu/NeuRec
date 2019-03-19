from evaluation.leaveoneout import HR
from evaluation.leaveoneout import NDCG as looNDCG
from evaluation.leaveoneout import AUC as looAUC
import heapq  # for retrieval topK
import numpy as np
from concurrent.futures import ThreadPoolExecutor  # xxxxxxxxxxxxxxxxxx


def evaluate_by_loo(model,evaluateMatrix,evaluateNegatives):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _trainMatrix
    global _evaluateMatrix
    global _evaluateNegatives
    global _K
    _model = model
    _trainMatrix = _model.dataset.trainMatrix.tocsr()
    _evaluateMatrix = evaluateMatrix.tocsr()
    _evaluateNegatives = evaluateNegatives
    _K = _model.topK
    num_thread = 20  # xxxxxxxxxxxxxxxxxx
    hits, ndcgs,aucs = [], [],[]
    if(num_thread > 1): # Multi-thread
        with ThreadPoolExecutor() as executor:  # xxxxxxxxxxxxxxxxxx
            res = executor.map(eval_by_loo_user, range(_evaluateMatrix.shape[0]))  # xxxxxxxxxxxxxxxxxx
        res = list(res)  # xxxxxxxxxxxxxxxxxx
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        aucs = [r[2] for r in res]
    # Single thread
    else:
       
        # Single thread
        for u in range(_model.num_users):
            if len(_evaluateMatrix[u].indices) !=0:
                (hr, ndcg,auc) = eval_by_loo_user(u)  # xxxxxxxxxxxxxxxxxx
                aucs.append(auc)
                hits.append(hr)
                ndcgs.append(ndcg)
            
    return (hits, ndcgs,aucs)

def eval_by_loo_user(u):  # xxxxxxxxxxxxxxxxxx
    target_item = _evaluateMatrix[u].indices[0]
    eval_items =[]
    if _evaluateNegatives != None:
        eval_items = _evaluateNegatives[u]
    else :
        all_items = set(np.arange(_model.num_items))
        eval_items = list(all_items - set(_trainMatrix[u].indices))
    eval_items.append(target_item)
    # Get prediction scores
    map_item_score = {}
    predictions = _model.predict(u,eval_items)
    for i in np.arange(len(eval_items)):
        item = eval_items[i]
        map_item_score[item] = predictions[i]
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = HR.getHitRatio(ranklist, target_item)
    ndcg = looNDCG.getNDCG(ranklist, target_item)
    auc = looAUC.getAUC(predictions, _K)
    return (hr,ndcg,auc)