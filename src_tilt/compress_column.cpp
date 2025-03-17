#include "compress_column.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include "Array.h"

void compress_columns(const int n_x, const int n_y, 
                      const int n_out, const int n_tilt, 
                      const int compress_lay_start_idx,
                      std::vector<Float>& var)
{
    std::vector<Float> var_tmp(n_out * n_x * n_y);    
    #pragma omp parallel for
    for (int ilay = 0; ilay < compress_lay_start_idx; ++ilay)
    {
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                var_tmp[out_idx] = var[out_idx];
            }
        }
    }

    #pragma omp parallel for
    for (int ilay = compress_lay_start_idx; ilay < n_out; ++ilay)
    {
        const int in_offset = ilay - compress_lay_start_idx;
        const int i_lay_in = compress_lay_start_idx + 2 * in_offset;
        int num_inputs;
        if (ilay < (n_out - 1)) {
            num_inputs = 2;
        } 
        else {
            num_inputs = ((i_lay_in + 1) == (n_tilt - 1)) ? 2 : 3;
        }
        
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                Float sum = 0.0;
                for (int k = 0; k < num_inputs; ++k)
                {
                    int in_idx = ix + iy * n_x + (i_lay_in + k) * n_x * n_y;
                    sum += var[in_idx];
                }
                var_tmp[out_idx] = sum;
            }
        }
    }
    var = var_tmp;
}

void compress_columns_weighted_avg(const int n_x, const int n_y,  
                      const int n_out, 
                      const int n_tilt,
                      const int compress_lay_start_idx,
                      std::vector<Float>& var, std::vector<Float>& var_weighting)
{
    std::vector<Float> var_tmp(n_out * n_x * n_y);

    #pragma omp parallel for
    for (int ilay = 0; ilay < compress_lay_start_idx; ++ilay)
    {
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                var_tmp[out_idx] = var[out_idx];
            }
        }
    }

    #pragma omp parallel for
    for (int ilay = compress_lay_start_idx; ilay < n_out; ++ilay)
    {
        const int in_offset = ilay - compress_lay_start_idx;
        const int i_lay_in = (compress_lay_start_idx + 2 * in_offset);
        int num_inputs;
        if (ilay < (n_out - 1)) {
            num_inputs = 2;
        } else {
            num_inputs = ((i_lay_in + 1) == (n_tilt - 1)) ? 2 : 3;
        }

        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                Float t_sum = 0.0;
                Float w_sum = 0.0;
                for (int k = 0; k < num_inputs; ++k)
                {
                    int in_idx = ix + iy * n_x + (i_lay_in + k) * n_x * n_y;
                    t_sum += var[in_idx] * var_weighting[in_idx];
                    w_sum += var_weighting[in_idx];
                }

                if (w_sum > 1e-6)
                {
                    var_tmp[out_idx] = t_sum / w_sum;
                } 
                else 
                {
                    Float avg = 0.0;
                    for (int k = 0; k < num_inputs; ++k)
                    {
                        int in_idx = ix + iy * n_x + (i_lay_in + k) * n_x * n_y;
                        avg += var[in_idx];
                    }
                    var_tmp[out_idx] = avg / num_inputs;
                }
            }
        }
    }
    var = var_tmp;
}

void compress_columns_p_or_t(const int n_x, const int n_y, 
                      const int n_out_lay,  const int n_tilt,
                      const int compress_lay_start_idx,
                      std::vector<Float>& var_lev, std::vector<Float>& var_lay)
{
    std::vector<Float> var_tmp_lay(n_out_lay * n_x * n_y);
    std::vector<Float> var_tmp_lev((n_out_lay + 1) * n_x * n_y);

    #pragma omp parallel for
    for (int ilay = 0; ilay < compress_lay_start_idx; ++ilay)
    {
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                var_tmp_lay[out_idx] = var_lev[out_idx];
                var_tmp_lev[out_idx] = var_lev[out_idx];
            }
        }
    }

    int ilay = compress_lay_start_idx;
    for (int iy = 0; iy < n_y; ++iy)
    {
        for (int ix = 0; ix < n_x; ++ix)
        {
            const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
            var_tmp_lev[out_idx] = var_lev[out_idx];
        }
    }

    #pragma omp parallel for
    for (int ilev = (compress_lay_start_idx + 1); ilev < (n_out_lay + 1); ++ilev)
    {
        int i_lev_in;
        if (ilev == n_out_lay)
        {
            i_lev_in = n_tilt;
        }
        else
        {
            i_lev_in = (compress_lay_start_idx + 2) + 2 * (ilev - (compress_lay_start_idx + 1));
        }

        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilev * n_x * n_y;
                const int in_idx = ix + iy * n_x + i_lev_in * n_x * n_y;
                var_tmp_lev[out_idx] = var_lev[in_idx];
            }
        }
    }

    #pragma omp parallel for
    for (int ilay = compress_lay_start_idx; ilay < n_out_lay; ++ilay)
    {
        const int in_offset = ilay - compress_lay_start_idx;
        int i_lev_to_lay_in;
        if (ilay == (n_out_lay - 1))
        {
            i_lev_to_lay_in = n_tilt - 1; // in some cases this is a slight approximation.
        }
        else 
        {
            i_lev_to_lay_in = (compress_lay_start_idx + 2 * in_offset - 1);
        }
        
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                const int in_idx_lev_to_lay = ix + iy * n_x + i_lev_to_lay_in * n_x * n_y;
                
                var_tmp_lay[out_idx] = var_lev[in_idx_lev_to_lay];
            }
        }
    }
    var_lev = var_tmp_lev;
    var_lay = var_tmp_lay;
}
