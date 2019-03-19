'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: wubin
'''
import numpy as np
from time import time
import logging
from evaluation.leaveoneout.LeaveOneOutEvaluate import evaluate_by_loo
from evaluation.foldout.FoldOutEvaluate import evaluate_by_foldout
_trainMatrix = None
_model = None
_testNegatives = None
_evaluateMatrix=None
_K = None

def test_model(model,dataset):
    eval_begin = time()
    model_name=str(model.__class__).split(sep=".")[-1].replace("\'>","")
    if dataset.splitter == "loo":
        (hits, ndcgs,aucs) = evaluate_by_loo(model,dataset.testMatrix,dataset.testNegatives)
        hr = np.array(hits).mean()
        ndcg = np.array(ndcgs).mean()
        auc = np.array(aucs).mean()
        logging.info(
            "[model=%s][loss_function=%s]: [Test HR = %.4f, NDCG = %.4f,AUC = %.4f] [Time=%.1fs]" % (model_name,model.loss_function,
            hr, ndcg,auc, time() - eval_begin))
        print ("[model=%s][loss_function=%s]: [Test HR = %.4f, NDCG = %.4f,AUC = %.4f] [Time=%.1fs]" % (model_name,model.loss_function,
            hr, ndcg,auc, time() - eval_begin))
        
    else:
        (pres,recs,maps,ndcgs,mrrs) = evaluate_by_foldout(model,dataset.testMatrix,dataset.testNegatives)
        Precision = np.array(pres).mean()
        Recall = np.array(recs).mean()
        MAP = np.array(maps).mean()
        NDCG = np.array(ndcgs).mean()
        MRR = np.array(mrrs).mean()    
        print ("[model=%s][loss_function=%s][%.1fs]: [Test Precision = %.4f, Recall= %.4f, MAP= %.4f, NDCG= %.4f, MRR= %.4f][topk=%.4s]"
               %(model_name,model.loss_function,time() - eval_begin, Precision, Recall,MAP,NDCG,MRR,model.topK))     