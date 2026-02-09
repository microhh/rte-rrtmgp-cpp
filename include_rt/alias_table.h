#ifndef ALIAS_TABLE_H
#define ALIAS_TABLE_H

#include "types.h"

#ifdef __CUDACC__
__device__ inline int sample_alias_table(
        const Float* __restrict__ prob,
        const int* __restrict__ alias,
        const int n,
        const Float u1,
        const Float u2)
{
    const int i = min(int(u1 * n), n-1);
    return (u2 < prob[i]) ? i : alias[i];
}
#endif

void build_alias_table(
        const Float* weights_gpu,
        const int n,
        Float* prob_gpu,
        int* alias_gpu,
        Float& total_sum);
#endif
