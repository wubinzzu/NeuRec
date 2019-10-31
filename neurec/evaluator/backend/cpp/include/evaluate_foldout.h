/*
@author: Zhongchuan Sun
*/
#ifndef EVALUATE_FOLDOUT_H
#define EVALUATE_FOLDOUT_H

#include <vector>
#include <set>
#include <cmath>
#include <future>
#include "thread_pool.h"

using std::vector;
using std::set;
using std::future;


vector<float> precision(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result(top_k);
    int hits = 0;
    set<int> truth_set(truth, truth+truth_len);
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            hits += 1;
        }
        result[i] = 1.0*hits / (i+1);
    }
    return result;
}

vector<float> recall(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result(top_k);
    int hits = 0;
    set<int> truth_set(truth, truth+truth_len);
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            hits += 1;
        }
        result[i] = 1.0*hits / truth_len;
    }
    return result;
}

vector<float> ap(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result(top_k); // = precision(rank, top_k, truth, truth_len);
    int hits = 0;
    float pre = 0;
    float sum_pre = 0;
    set<int> truth_set(truth, truth+truth_len);
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            hits += 1;
            pre = 1.0*hits / (i+1);
            sum_pre += pre;
        }
        result[i] = sum_pre/truth_len;
    }
    return result;
}

vector<float> ndcg(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result(top_k);
    float iDCG = 0;
    float DCG = 0;
    set<int> truth_set(truth, truth+truth_len);
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            DCG += 1.0/log2(i+2);
        }
        iDCG += 1.0/log2(i+2);
        result[i] = DCG/iDCG;
    }
    return result;
}

vector<float> mrr(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result(top_k);
    float rr = 0;
    set<int> truth_set(truth, truth+truth_len);
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            rr = 1.0/(i+1);
            for(int j=i; j<top_k; j++)
            {
                result[j] = rr;
            }
            break;
        }
        else
        {
            rr = 0.0;
            result[i] =rr;
        }
    }
    return result;
}


void evaluate_foldout(int users_num,
                      int *rankings, int rank_len,
                      int **ground_truths, int *ground_truths_num,
                      int thread_num, float *results)
{
    // typeassert();
    ThreadPool pool(thread_num);
    vector< future< vector<float> > > sync_pre_results;
    vector< future< vector<float> > > sync_recall_results;
    vector< future< vector<float> > > sync_ap_results;
    vector< future< vector<float> > > sync_ndcg_results;
    vector< future< vector<float> > > sync_mrr_results;
    
    for(int uid=0; uid<users_num; uid++)
    {
        int *cur_rankings = rankings + uid*rank_len;
        sync_pre_results.emplace_back(pool.enqueue(precision, cur_rankings, rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_recall_results.emplace_back(pool.enqueue(recall, cur_rankings, rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_ap_results.emplace_back(pool.enqueue(ap, cur_rankings, rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_ndcg_results.emplace_back(pool.enqueue(ndcg, cur_rankings, rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_mrr_results.emplace_back(pool.enqueue(mrr, cur_rankings, rank_len, ground_truths[uid], ground_truths_num[uid]));
    }
    
    float *pre_offset = results + 0*rank_len;  // the offset address of precision in the first user result
    float *recall_offset = results + 1*rank_len;  // the offset address of recall in the first user result
    float *ap_offset = results + 2*rank_len;  // the offset address of map in the first user result
    float *ndcg_offset = results + 3*rank_len;  // the offset address of ndcg in the first user result
    float *mrr_offset = results + 4*rank_len;  // the offset address of mrr in the first user result
    
    int metric_num = 5;

    for(auto && result: sync_pre_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            pre_offset[k] = tmp_result[k];
        }
        pre_offset += rank_len*metric_num;  // move to the next user's result address
    }

    for(auto && result: sync_recall_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            recall_offset[k] = tmp_result[k];
        }
        recall_offset += rank_len*metric_num;  // move to the next user's result address
    }

    for(auto && result: sync_ap_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            ap_offset[k] = tmp_result[k];
        }
        ap_offset += rank_len*metric_num;  // move to the next user's result address
    }

    for(auto && result: sync_ndcg_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            ndcg_offset[k] = tmp_result[k];
        }
        ndcg_offset += rank_len*metric_num;  // move to the next user's result address
    }

    for(auto && result: sync_mrr_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            mrr_offset[k] = tmp_result[k];
        }
        mrr_offset += rank_len*metric_num;  // move to the next user's result address
    }
}

#endif