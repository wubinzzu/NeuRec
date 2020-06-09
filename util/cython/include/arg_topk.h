/*
@author: Zhongchuan Sun
*/
#ifndef ARG_TOPK_H
#define ARG_TOPK_H

#include "thread_pool.h"
#include <vector>
#include <algorithm>
#include <future>
using std::vector;
using std::future;


int arg_top_k_1d(float *ratings, int rating_len, int top_k, int *result)
{
    vector<int> index(rating_len);
    for(auto i=0; i<rating_len; ++i)
    {
        index[i] = i;
    }
    std::partial_sort_copy(index.begin(), index.end(), result, result+top_k,
                            [& ratings](int &x1, int &x2)->bool{return ratings[x1]>ratings[x2];});
    return 0;
}



void arg_top_k_2d(float *ratings, int rating_len, int rows_num, int top_k, int thread_num, int *results_pt)
{
    ThreadPool pool(thread_num);
    vector< future< int > > sync_results;

    for(int i=0; i<rows_num; ++i)
    {
        auto rating_pt = ratings + i*rating_len;
        auto r_pt = results_pt + i*top_k;
        sync_results.emplace_back(pool.enqueue(arg_top_k_1d, rating_pt, rating_len, top_k, r_pt));
    }

    for(auto && result: sync_results)
    {
        result.get();  // join
    }
}

#endif

