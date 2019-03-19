from evaluation.foldout import Precision 
from evaluation.foldout import Recall 
from evaluation.foldout import MAP 
from evaluation.foldout import MRR 
from evaluation.foldout import NDCG
import heapq  # for retrieval topK
import numpy as np
from concurrent.futures import ThreadPoolExecutor  # xxxxxxxxxxxxxxxxxx


def evaluate_by_foldout(model,evaluateMatrix,evaluateNegatives):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _trainMatrix
    global _testMatrix
    global _evaluateMatrix
    global _evaluateNegatives
    global _K
    global _evaluateusers
    _model = model
    _trainMatrix = _model.dataset.trainMatrix.tocsr()
    _testMatrix = _model.dataset.testMatrix.tocsr()
    _evaluateMatrix = evaluateMatrix.tocsr()
    _evaluateNegatives = evaluateNegatives
    _evaluateusers = []
    _K = _model.topK
    num_thread = 10  # xxxxxxxxxxxxxxxxxx
    Pres, Recs,MAPs,NDCGs,MRRs = [],[],[],[],[]
    for u in range(_model.num_users):
        items_test = _testMatrix[u].indices
        if len(items_test) >0:
            _evaluateusers.append(u)
    if(num_thread > 1): # Multi-thread
        with ThreadPoolExecutor() as executor:  # xxxxxxxxxxxxxxxxxx
            res = executor.map(eval_by_foldout_user, _evaluateusers)  # xxxxxxxxxxxxxxxxxx
        #res = pool.map(eval_by_foldout_user, range(len(_evaluateMatrix)))
        #pool.close()
        #pool.join()
        res = list(res)  # xxxxxxxxxxxxxxxxxx
        Pres = [r[0] for r in res]
        Recs = [r[1] for r in res]
        MAPs = [r[2] for r in res]
        NDCGs = [r[3] for r in res]
        MRRs = [r[4] for r in res]
    # Single thread
    else:
        # Single thread
        for u in _evaluateusers:
            if len(_evaluateMatrix[u].indices) !=0:
                (Pre,Rec,MAP,NDCG,MRR) = eval_by_foldout_user(u)  # xxxxxxxxxxxxxxxxxx
                Pres.append(Pre) 
                Recs.append(Rec)
                MAPs.append(MAP)
                NDCGs.append(NDCG)
                MRRs.append(MRR)
    return (Pres,Recs,MAPs,NDCGs,MRRs)
def eval_by_foldout_user(u):  # xxxxxxxxxxxxxxxxxx
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
    predictions = _model.predict(u,eval_items)
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

