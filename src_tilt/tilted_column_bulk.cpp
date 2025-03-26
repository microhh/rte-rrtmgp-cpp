#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include "Source_functions.h"
#include "Gas_concs.h"
#include "tilted_column.h"
#include "tilted_column_bulk.h"

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

void tilt_fields(const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1> zh, const Array<Float,1> z,
    const Array<Float,1> zh_tilt, const Array<ijk,1> path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Gas_concs& gas_concs_copy, const std::vector<std::string> gas_names
) {
    for (const auto& gas_name : gas_names) {
        if (!gas_concs_copy.exists(gas_name)) {
            continue;
        }
        const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);

        if (gas.size() > 1) {
            Array<Float,2> gas_tmp(gas);
            create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), gas_tmp.v());
            gas_tmp.expand_dims({n_col, n_z_tilt});
            gas_concs_copy.set_vmr(gas_name, gas_tmp);
        } 
        else if (gas.size() == 1) {
            // Do nothing for single profiles
        } 
        else {
            throw std::runtime_error("No tilted column implementation for single profiles.");
        }
    }

    // Create tilted columns for T and p. Important: create T first!!
    create_tilted_columns_levlay(n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), t_lay_copy->v(), t_lev_copy->v());
    create_tilted_columns_levlay(n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), p_lay_copy->v(), p_lev_copy->v());

    t_lay_copy->expand_dims({n_col, n_z_tilt});
    t_lev_copy->expand_dims({n_col, n_zh_tilt});
    p_lay_copy->expand_dims({n_col, n_z_tilt});
    p_lev_copy->expand_dims({n_col, n_zh_tilt});
}

void compress_fields(const int compress_lay_start_idx, const int n_col_x, const int n_col_y,
    const int n_z_in, const int n_zh_in,  const int n_z_tilt,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Gas_concs& gas_concs_copy, std::vector<std::string> gas_names)
{
    const int n_col = n_col_x*n_col_y;

    for (const auto& gas_name : gas_names) {
        if (!gas_concs_copy.exists(gas_name)) {
            continue;
        }
        const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);
        if (gas.size() > 1) {
            Array<Float,2> gas_tmp(gas);
            compress_columns_weighted_avg(n_col_x, n_col_y,
                                            n_z_in, n_z_tilt,
                                            compress_lay_start_idx, 
                                            gas_tmp.v(), 
                                            p_lay_copy->v());
            gas_tmp.expand_dims({n_col, n_z_in});
            gas_concs_copy.set_vmr(gas_name, gas_tmp);
        }
    }

    compress_columns_p_or_t(n_col_x, n_col_y, n_z_in, n_z_tilt,
                            compress_lay_start_idx, 
                            p_lev_copy->v(), p_lay_copy->v());
    p_lay_copy->expand_dims({n_col, n_z_in});
    p_lev_copy->expand_dims({n_col, n_zh_in});
    compress_columns_p_or_t(n_col_x, n_col_y, n_z_in, n_z_tilt,
                            compress_lay_start_idx, 
                            t_lev_copy->v(), t_lay_copy->v());
    t_lay_copy->expand_dims({n_col, n_z_in});
    t_lev_copy->expand_dims({n_col, n_zh_in});
}

void create_tilted_columns(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                           const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                           std::vector<Float>& var)
{
    const int n_lay = tilted_path.size();
    const int n_lev = zh_tilted.size();

    std::vector<Float> var_tmp(n_lay*n_x*n_y);

    #pragma omp parallel for
    for (int ilay=0; ilay<n_lay; ++ilay)
    {
        const ijk offset = tilted_path[ilay];
        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y + ilay*n_y*n_x;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + offset.k*n_y*n_x;
                var_tmp[idx_out] = var[idx_in];
            } 
    }

    var.resize(var_tmp.size());
    var = var_tmp;
}

void interpolate(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                 const std::vector<Float>& zh_in, const std::vector<Float>& zf_in,
                 const std::vector<Float>& play_in, const std::vector<Float>& plev_in,
                 const Float zp, const ijk offset,
                 Float* p_out)
{
    int zp_in_zh = -1; // half level
    int zp_in_zf = -1; // full level
    for (int ilev=0; ilev<n_lev_in; ++ilev)
        if (std::abs(zh_in[ilev]-zp) < 1e-2)
            zp_in_zh= ilev;
    for (int ilay=0; ilay<n_lay_in; ++ilay)
        if (std::abs(zf_in[ilay]-zp) < 1e-2)
            zp_in_zf = ilay;
    if (zp_in_zh > -1)
    {
        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + zp_in_zh*n_y*n_x;
                p_out[idx_out] = plev_in[idx_in];
            } 
    }
    else if (zp_in_zf > -1)
    {
        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + zp_in_zf*n_y*n_x;
                p_out[idx_out] = play_in[idx_in];
            } 
    }
    else
    {
        int posh_bot = 0;
        int posf_bot = 0;
        for (int ilev=0; ilev<n_lev_in-1; ++ilev)
            if (zh_in[ilev] < zp)
                posh_bot = ilev;     
        for (int ilay=0; ilay<n_lay_in; ++ilay)
            if (zf_in[ilay] < zp)
                posf_bot = ilay;         
        const Float* p_top;
        const Float* p_bot;
        Float  z_top;
        Float  z_bot;

        const int zh_top = zh_in[posh_bot+1];
        const int zh_bot = zh_in[posh_bot];
        const int zf_top = (posf_bot+1 < n_lay_in) ? zf_in[posf_bot+1] : zh_top+1;
        const int zf_bot = zf_in[posf_bot];

        if (zh_top > zf_top)
        {
            p_top = &play_in.data()[(posf_bot+1)*n_x*n_y];       
            z_top = zf_in[posf_bot+1];       
        }
        else
        {   
            p_top = &plev_in.data()[(posh_bot+1)*n_x*n_y];       
            z_top = zh_in[posh_bot+1];
        }
        if (zh_bot < zf_bot)
        {
            p_bot = &play_in.data()[(posf_bot)*n_x*n_y];       
            z_bot = zf_in[posf_bot];       
        }
        else
        {   
            p_bot = &plev_in.data()[(posh_bot)*n_x*n_y];       
            z_bot = zh_in[posh_bot];
        }

        Float dz = z_top-z_bot;

        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x;
                const Float pres = (zp-z_bot)/dz*p_top[idx_in] + (z_top-zp)/dz*p_bot[idx_in];
                p_out[idx_out] = pres;
            } 

    }
}


void create_tilted_columns_levlay(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                                 const std::vector<Float>& zh_in, const std::vector<Float>& z_in,
                                 const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                                 std::vector<Float>& var_lay, std::vector<Float>& var_lev)

{
    const int n_lay = tilted_path.size();
    const int n_lev = zh_tilted.size();
    std::vector<Float> z_tilted(n_lay);
    for (int ilay=0; ilay<n_lay; ++ilay)
        z_tilted[ilay] = (zh_tilted[ilay]+zh_tilted[ilay+1])/Float(2.);

    std::vector<Float> var_lay_tmp(n_lay*n_x*n_y);
    std::vector<Float> var_lev_tmp(n_lev*n_x*n_y);
    for (int iy=0; iy<n_y; ++iy)
        for (int ix=0; ix<n_x; ++ix)
        {
            const int idx = ix + iy*n_y;
            var_lev_tmp[idx] = var_lev[idx];
        } 

    #pragma omp parallel for
    for (int ilev=1; ilev<n_lev; ++ilev)
        interpolate(n_x, n_y, n_lay_in, n_lev_in, zh_in, z_in, var_lay, var_lev, zh_tilted[ilev], tilted_path[ilev-1], &var_lev_tmp.data()[ilev*n_y*n_x]);
    #pragma omp parallel for
    for (int ilay=0; ilay<n_lay; ++ilay)
        interpolate(n_x, n_y, n_lay_in, n_lev_in, zh_in, z_in, var_lay, var_lev, z_tilted[ilay], tilted_path[ilay], &var_lay_tmp.data()[ilay*n_y*n_x]);

    var_lay.resize(n_lay*n_y*n_x);
    var_lev.resize(n_lev*n_y*n_x);
    var_lay = var_lay_tmp;
    var_lev = var_lev_tmp;
}
