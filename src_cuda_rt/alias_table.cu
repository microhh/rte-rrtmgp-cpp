#include <cub/cub.cuh>

#include "alias_table.h"
#include "tools_gpu.h"

namespace
{
    __global__
    void normalize_and_init(
            const Float* __restrict__ weights,
            const int n,
            const double total_sum,
            double* __restrict__ w,
            double* __restrict__ prob,
            int* __restrict__ alias,
            int* __restrict__ active)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            w[i] = static_cast<double>(weights[i]) * double(n) / total_sum;
            prob[i] = 1.0;
            alias[i] = i;
            active[i] = i;
        }
    }


    __global__
    void pair_entries(
            const int* __restrict__ small_indices,
            const int* __restrict__ large_indices,
            const int k,
            double* __restrict__ w,
            double* __restrict__ prob,
            int* __restrict__ alias)
    {
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j < k)
        {
            const int s = small_indices[j];
            const int l = large_indices[j];
            prob[s] = w[s];
            alias[s] = l;
            w[l] += w[s] - 1.0;
        }
    }


    struct Is_small
    {
        const double* w;
        __device__ __forceinline__ bool operator()(int idx) const
        {
            return w[idx] < 1.0;
        }
    };
}


void build_alias_table(
        const Float* weights_gpu,
        const int n,
        double* prob_gpu,
        int* alias_gpu,
        double& total_sum)
{
    if (n == 0) return;

    // 1. Reduce to get total sum.
    double* d_total = Tools_gpu::allocate_gpu<double>(1);

    void* d_reduce_temp = nullptr;
    size_t reduce_temp_bytes = 0;
    cub::DeviceReduce::Sum(d_reduce_temp, reduce_temp_bytes, weights_gpu, d_total, n);
    cudaMalloc(&d_reduce_temp, reduce_temp_bytes);
    cub::DeviceReduce::Sum(d_reduce_temp, reduce_temp_bytes, weights_gpu, d_total, n);
    cudaFree(d_reduce_temp);

    cuda_safe_call(cudaMemcpy(&total_sum, d_total, sizeof(double), cudaMemcpyDeviceToHost));
    Tools_gpu::free_gpu(d_total);

    if (total_sum <= 0.) return;

    // 2. Allocate working arrays.
    double* w = Tools_gpu::allocate_gpu<double>(n);
    int* buf_a = Tools_gpu::allocate_gpu<int>(n);
    int* buf_b = Tools_gpu::allocate_gpu<int>(n);
    int* d_n_small = Tools_gpu::allocate_gpu<int>(1);

    // 3. Normalize weights to sum to n, initialize prob and alias.
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    normalize_and_init<<<grid_size, block_size>>>(
            weights_gpu, n, total_sum, w, prob_gpu, alias_gpu, buf_a);

    // 4. Iterative partition and pair.
    Is_small pred{w};

    void* d_part_temp = nullptr;
    size_t part_temp_bytes = 0;
    cub::DevicePartition::If(
            d_part_temp, part_temp_bytes,
            buf_a, buf_b, d_n_small, n, pred);
    cudaMalloc(&d_part_temp, part_temp_bytes);

    int n_active = n;

    while (n_active > 0)
    {
        size_t temp_bytes = part_temp_bytes;
        cub::DevicePartition::If(
                d_part_temp, temp_bytes,
                buf_a, buf_b, d_n_small, n_active, pred);

        int n_small;
        cuda_safe_call(cudaMemcpy(&n_small, d_n_small, sizeof(int), cudaMemcpyDeviceToHost));

        const int n_large = n_active - n_small;
        if (n_small == 0 || n_large == 0) break;

        const int k = min(n_small, n_large);
        const int pair_grid = (k + block_size - 1) / block_size;

        pair_entries<<<pair_grid, block_size>>>(
                buf_b, buf_b + n_small, k, w, prob_gpu, alias_gpu);

        // Carry forward unpaired small entries and all (adjusted) large entries.
        const int unpaired_small = n_small - k;
        if (unpaired_small > 0)
            cuda_safe_call(cudaMemcpy(
                    buf_a, buf_b + k, unpaired_small * sizeof(int), cudaMemcpyDeviceToDevice));
        cuda_safe_call(cudaMemcpy(
                buf_a + unpaired_small, buf_b + n_small, n_large * sizeof(int), cudaMemcpyDeviceToDevice));
        n_active = unpaired_small + n_large;
    }

    cudaFree(d_part_temp);
    Tools_gpu::free_gpu(w);
    Tools_gpu::free_gpu(buf_a);
    Tools_gpu::free_gpu(buf_b);
    Tools_gpu::free_gpu(d_n_small);
}
