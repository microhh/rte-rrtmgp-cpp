#include <chrono>
#include <iomanip>

#include "subset_kernel_launcher_cuda_rt.h"
#include "tools_gpu.h"
#include "array.h"

namespace
{
    #include "subset_kernels_rt.cu"
}

namespace subset_kernel_launcher_cuda_rt
{
    
    void get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
                  Float* var1_full, Float* var2_full, Float* var3_full,  Float* var4_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub, const Float* var4_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ncol_in, col_s_in, var1_full, var2_full,
                var3_full, var4_full, var1_sub, var2_sub,
                var3_sub, var4_sub);
    }

    
    void get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
                  Float* var1_full, Float* var2_full, Float* var3_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ncol_in, col_s_in, var1_full, var2_full,
                var3_full, var1_sub, var2_sub,
                var3_sub);
    }

    
    void get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  Float* var1_full, Float* var2_full, Float* var3_full,  Float* var4_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub, const Float* var4_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;
        const int block_bnd = 1;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
        dim3 block_gpu(block_col, block_lay, block_bnd);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ncol_in, col_s_in, var1_full, var2_full,
                var3_full, var4_full, var1_sub, var2_sub,
                var3_sub, var4_sub);
    }

    
    void get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  Float* var1_full, Float* var2_full, Float* var3_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;
        const int block_bnd = 1;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
        dim3 block_gpu(block_col, block_lay, block_bnd);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ncol_in, col_s_in, var1_full, var2_full,
                var3_full, var1_sub, var2_sub, var3_sub);
    }

    
    void get_from_subset(const int ncol, const int nbnd, const int ncol_in, const int col_s_in,
                  Float* var_full, const Float* var_sub)
    {
        const int block_col = 16;
        const int block_bnd = 14;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_bnd);
        dim3 block_gpu(block_col, block_bnd);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(ncol, nbnd, ncol_in, col_s_in, var_full, var_sub);
    }
}

