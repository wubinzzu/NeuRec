/*
@author: Zhongchuan Sun
*/
#ifndef TOOLS_H
#define TOOLS_H

#include "thread_pool.h"
#include <vector>
#include <algorithm>
using std::vector;


void c_top_k_index(float *ratings, int rating_len, int top_k, int *result)
{
    vector<int> index(rating_len);
    for(auto i=0; i<rating_len; ++i)
    {
        index[i] = i;
    }
    std::partial_sort_copy(index.begin(), index.end(), result, result+top_k,
                            [& ratings](int &x1, int &x2)->bool{return ratings[x1]>ratings[x2];});
}

void c_top_k_array_index(float *scores_pt, int columns_num, int rows_num, int top_k, int thread_num, int *rankings_pt)
{
    ThreadPool pool(thread_num);
    for(int i=0; i<rows_num; ++i)
    {
        float *cur_scores_pt = scores_pt + columns_num*i;
        int *cur_ranking_pt = rankings_pt + top_k*i;
        pool.enqueue(c_top_k_index, cur_scores_pt, columns_num, top_k, cur_ranking_pt);
    }
}

#endif