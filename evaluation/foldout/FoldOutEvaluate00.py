from evaluation.foldout import Precision 
from evaluation.foldout import Recall 
from evaluation.foldout import MAP 
from evaluation.foldout import MRR 
from evaluation.foldout import NDCG 
import multiprocessing
import heapq  # for retrieval topK
import numpy as np
def evaluate_by_foldout(model,evaluateMatrix,evaluateNegatives,isvalid):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _trainMatrix
    global _evaluateMatrix
    global _evaluateNegatives
    global _K
    global _isvalid
    _model = model
    _trainMatrix = _model.dataset.trainMatrix.tocsr()
    _evaluateMatrix = evaluateMatrix.tocsr()
    _evaluateNegatives = evaluateNegatives
    _isvalid = isvalid
    _K = _model.topK
    num_thread = 1
    Pres, Recs,MAPs,NDCGs,MRRs = [],[],[],[],[]
    if(num_thread > 1): # Multi-thread
        multiprocessing.freeze_support()
        pool = multiprocessing.Pool(num_thread)
        res = pool.map(eval_by_foldout_user, range(len(_evaluateMatrix)))
        pool.close()
        pool.join()
        Pres = [r[0] for r in res]
        Recs = [r[1] for r in res]
        MAPs = [r[2] for r in res]
        NDCGs = [r[3] for r in res]
        MRRs = [r[4] for r in res]
    # Single thread
    else:
        # Single thread
        for u in range(_model.num_users):
            if len(_evaluateMatrix[u].indices) !=0:
                (Pre,Rec,MAP,NDCG,MRR) = eval_by_foldout_user(u,_isvalid)
                Pres.append(Pre) 
                Recs.append(Rec)
                MAPs.append(MAP)
                NDCGs.append(NDCG)
                MRRs.append(MRR)
    return (Pres,Recs,MAPs,NDCGs,MRRs)
def eval_by_foldout_user(u,_isvalid):
    target_items= _evaluateMatrix[u].indices
    eval_items =[]
    if _evaluateNegatives != None:
        eval_items = _evaluateNegatives[u]
    else :
        all_items = set(np.arange(_model.num_items))
        eval_items = list(all_items - set(_trainMatrix[u].indices))
    eval_items.extend(target_items)
    # Get prediction scores
    map_item_score = {}
    predictions = _model.predict(u,eval_items,_isvalid)
    for i in np.arange(len(eval_items)):
        item = eval_items[i]
        map_item_score[item] = predictions[i]
    # Evaluate top rank list
    rank_list = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    target_items = set(target_items)
    Pre = Precision.getPre(rank_list, target_items,_K)
    Rec = Recall.getRec(rank_list, target_items)
    ap = MAP.getAP(rank_list, target_items)
    dcg = NDCG.getNDCG(rank_list, target_items)
    rr = MRR.getMRR(rank_list, target_items)
    return (Pre,Rec,ap,dcg,rr)

