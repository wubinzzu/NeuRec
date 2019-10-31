/*
@author: Zhongchuan Sun
*/
#ifndef EVALUATE_LOO_H
#define EVALUATE_LOO_H

#include <vector>
#include <set>
#include <cmath>
#include <future>
#include "thread_pool.h"

using std::vector;
using std::set;
using std::future;


vector<float> hit(int *rank, int top_k, int truth)
{
    vector<float> result(top_k);
    int i = 0;
    for(i=0; i<top_k; i++)
    {
        if(rank[i] == truth)
        {
            result[i] = 1;
            break;
        }
    }
    for(; i<top_k; i++)
    {
        result[i] = 1;
    }
    return result;
}


vector<float> ndcg(int *rank, int top_k, int truth)
{
    vector<float> result(top_k);
    int i = 0;
    float DCG = 0;
    for(i=0; i<top_k; i++)
    {
        if(rank[i] == truth)
        {
            DCG = 1.0/log2(i+2);
            result[i] = DCG;
            break;
        }
    }
    for(; i<top_k; i++)
    {
        result[i] = DCG;
    }
    return result;
}

vector<float> mrr(int *rank, int top_k, int truth)
{
    vector<float> result(top_k);
    int i = 0;
    float rr = 0;
    for(i=0; i<top_k; i++)
    {
        if(rank[i] == truth)
        {
            rr = 1.0/(i+1);
            result[i] = rr;
            break;
        }
    }
    for(; i<top_k; i++)
    {
        result[i] = rr;
    }
    return result;
}


void evaluate_loo(int users_num, int *rankings, int rank_len,
                  int *ground_truths, int thread_num, float *results)
{
    ThreadPool pool(thread_num);
    vector< future< vector<float> > > sync_hit_results;
    vector< future< vector<float> > > sync_ndcg_results;
    vector< future< vector<float> > > sync_mrr_results;

    for(int uid=0; uid<users_num; uid++)
    {
        int *cur_rankings = rankings + uid*rank_len;
        sync_hit_results.emplace_back(pool.enqueue(hit, cur_rankings, rank_len, ground_truths[uid]));
        sync_ndcg_results.emplace_back(pool.enqueue(ndcg, cur_rankings, rank_len, ground_truths[uid]));
        sync_mrr_results.emplace_back(pool.enqueue(mrr, cur_rankings, rank_len, ground_truths[uid]));
    }

    float *hit_offset = results + 0*rank_len;  // the offset address of hit ratio in the first user result
    float *ndcg_offset = results + 1*rank_len;  // the offset address of ndcg in the first user result
    float *mrr_offset = results + 2*rank_len;  // the offset address of mrr in the first user result

    int metric_num = 3;

    for(auto && result: sync_hit_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            hit_offset[k] = tmp_result[k];
        }
        hit_offset += rank_len*metric_num;  // move to the next user's result address
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