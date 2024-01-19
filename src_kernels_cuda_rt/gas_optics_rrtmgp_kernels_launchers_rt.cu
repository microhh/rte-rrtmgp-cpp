#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>

#include "gas_optics_rrtmgp_kernels_cuda_rt.h"
#include "tools_gpu.h"
#include "tuner.h"


namespace
{
    #include "gas_optics_rrtmgp_kernels_rt.cu"

    using Tools_gpu::calc_grid_size;
}


namespace Gas_optics_rrtmgp_kernels_cuda_rt
{
    void reorder123x321(
            const int ni, const int nj, const int nk,
            const Float* arr_in, Float* arr_out)
    {
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid(ni, nj, nk);
        dim3 block;

        if (tunings.count("reorder123x321_kernel_rt") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "reorder123x321_kernel_rt",
                dim3(ni, nj, nk),
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                reorder123x321_kernel,
                ni, nj, nk, arr_in, arr_out);

            tunings["reorder123x321_kernel_rt"].first = grid;
            tunings["reorder123x321_kernel_rt"].second = block;
        }
        else
        {
            block = tunings["reorder123x321_kernel_rt"].second;
        }

        grid = calc_grid_size(block, dim3(ni, nj, nk));

        reorder123x321_kernel<<<grid, block>>>(
                ni, nj, nk, arr_in, arr_out);
    }


    void reorder12x21(const int ni, const int nj,
                      const Float* arr_in, Float* arr_out)
    {
        const int block_i = 32;
        const int block_j = 16;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);

        dim3 grid_gpu(grid_i, grid_j);
        dim3 block_gpu(block_i, block_j);

        reorder12x21_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, arr_in, arr_out);
    }


    void zero_array(const int ni, const int nj, const int nk, const int nn, Float* arr)
    {
        const int block_i = 32;
        const int block_j = 16;
        const int block_k = 1;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);
        const int grid_k = nk/block_k + (nk%block_k > 0);

        dim3 grid_gpu(grid_i, grid_j, grid_k);
        dim3 block_gpu(block_i, block_j, block_k);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, nk, nn, arr);
    }


    void zero_array(const int ni, const int nj, const int nk, Float* arr)
    {
        const int block_i = 32;
        const int block_j = 16;
        const int block_k = 1;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);
        const int grid_k = nk/block_k + (nk%block_k > 0);

        dim3 grid_gpu(grid_i, grid_j, grid_k);
        dim3 block_gpu(block_i, block_j, block_k);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, nk, arr);

    }


    void zero_array(const int ni, const int nj, Float* arr)
    {
        const int block_i = 32;
        const int block_j = 16;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);

        dim3 grid_gpu(grid_i, grid_j, 1);
        dim3 block_gpu(block_i, block_j, 1);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, arr);

    }

    void zero_array(const int ni, int* arr)
    {
        const int block_i = 32;

        const int grid_i = ni/block_i + (ni%block_i > 0);

        dim3 grid_gpu(grid_i);
        dim3 block_gpu(block_i);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, arr);
    }

    void interpolation(
            const int col_s, const int ncol_sub, const int ncol, const int nlay, const int igpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* gpoint_flavor,
            const int* flavor,
            const Float* press_ref_log,
            const Float* temp_ref,
            Float press_ref_log_delta,
            Float temp_ref_min,
            Float temp_ref_delta,
            Float press_ref_trop_log,
            const Float* vmr_ref,
            const Float* play,
            const Float* tlay,
            Float* col_gas,
            int* jtemp,
            Float* fmajor, Float* fminor,
            Float* col_mix,
            Bool* tropo,
            int* jeta,
            int* jpress)
    {
        Tuner_map& tunings = Tuner::get_map();
        Float tmin = std::numeric_limits<Float>::min();

        dim3 grid(nlay, ncol_sub, 1), block;
        if (tunings.count("interpolation_kernel_rt") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                    "interpolation_kernel_rt",
                    dim3(nlay, ncol_sub, 1),
                    {1,2,4}, {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, {1},
                    interpolation_kernel,
                    igpt-1, col_s, ncol_sub, ncol, nlay, ngas, nflav, neta, npres, ntemp, tmin,
                    gpoint_flavor, flavor, press_ref_log, temp_ref,
                    press_ref_log_delta, temp_ref_min,
                    temp_ref_delta, press_ref_trop_log,
                    vmr_ref, play, tlay,
                    col_gas, jtemp, fmajor,
                    fminor, col_mix, tropo,
                    jeta, jpress);
            tunings["interpolation_kernel_rt"].first = grid;
            tunings["interpolation_kernel_rt"].second = block;
        }
        else
        {
            block = tunings["interpolation_kernel_rt"].second;
        }

        grid = calc_grid_size(block, dim3(nlay, ncol_sub, 1));

        interpolation_kernel<<<grid, block>>>(
                igpt-1, col_s, ncol_sub, ncol, nlay, ngas, nflav, neta, npres, ntemp, tmin,
                gpoint_flavor, flavor, press_ref_log, temp_ref,
                press_ref_log_delta, temp_ref_min,
                temp_ref_delta, press_ref_trop_log,
                vmr_ref, play, tlay,
                col_gas, jtemp, fmajor,
                fminor, col_mix, tropo,
                jeta, jpress);

    }

    void combine_abs_and_rayleigh(
            const int col_s, const int ncol_sub, const int ncol, const int nlay,
            const Float* tau_abs, const Float* tau_rayleigh,
            Float* tau, Float* ssa, Float* g)
    {
        Tuner_map& tunings = Tuner::get_map();

        Float tmin = std::numeric_limits<Float>::epsilon();

        dim3 grid(ncol_sub, nlay, 1);
        dim3 block;

        if (tunings.count("combine_abs_and_rayleigh_kernel_rt") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "combine_abs_and_rayleigh_kernel_rt",
                dim3(ncol_sub, nlay, 1),
                {24, 32, 48, 64, 96, 128, 256, 512}, {1, 2, 4}, {1},
                combine_abs_and_rayleigh_kernel,
                col_s, ncol_sub, ncol, nlay, tmin,
                tau_abs, tau_rayleigh,
                tau, ssa, g);

            tunings["combine_abs_and_rayleigh_kernel_rt"].first = grid;
            tunings["combine_abs_and_rayleigh_kernel_rt"].second = block;
        }
        else
        {
            block = tunings["combine_abs_and_rayleigh_kernel_rt"].second;
        }

        grid = calc_grid_size(block, dim3(ncol_sub, nlay, 1));

        combine_abs_and_rayleigh_kernel<<<grid, block>>>(
                col_s, ncol_sub, ncol, nlay, tmin,
                tau_abs, tau_rayleigh,
                tau, ssa, g);
    }


    void compute_tau_rayleigh(
            const int col_s, const int ncol_sub, const int ncol, const int nlay, const int nbnd, const int ngpt, 
            const int igpt, const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* gpoint_bands,
            const int* band_lims_gpt,
            const Float* krayl,
            int idx_h2o, const Float* col_dry, const Float* col_gas,
            const Float* fminor, const int* jeta,
            const Bool* tropo, const int* jtemp,
            Float* tau_rayleigh)
    {
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid(ncol_sub, nlay, 1), block;
        if (tunings.count("compute_tau_rayleigh_kernel_rt") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "compute_tau_rayleigh_kernel_rt",
                dim3(ncol_sub, nlay, 1),
                {24, 32, 64, 128, 256, 512}, {1, 2}, {1},
                compute_tau_rayleigh_kernel,
                col_s, ncol_sub, ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                igpt-1,
                gpoint_bands,
                band_lims_gpt,
                krayl,
                idx_h2o, col_dry, col_gas,
                fminor, jeta,
                tropo, jtemp,
                tau_rayleigh);

            tunings["compute_tau_rayleigh_kernel_rt"].first = grid;
            tunings["compute_tau_rayleigh_kernel_rt"].second = block;
        }
        else
        {
            block = tunings["compute_tau_rayleigh_kernel_rt"].second;
        }

        grid = calc_grid_size(block, dim3(ncol_sub, nlay, 1));

        compute_tau_rayleigh_kernel<<<grid, block>>>(
                col_s, ncol_sub, ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                igpt-1,
                gpoint_bands,
                band_lims_gpt,
                krayl,
                idx_h2o, col_dry, col_gas,
                fminor, jeta,
                tropo, jtemp,
                tau_rayleigh);
    }


    void compute_tau_absorption(
            const int col_s, const int ncol_sub, const int ncol, const int nlay, const int nband, const int ngpt, 
            const int igpt, const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int nminorlower, const int nminorklower,
            const int nminorupper, const int nminorkupper,
            const int idx_h2o,
            const int* band_lims_gpt,
            const Float* kmajor,
            const Float* kminor_lower,
            const Float* kminor_upper,
            const int* minor_limits_gpt_lower,
            const int* minor_limits_gpt_upper,
            const int* first_last_minor_lower,
            const int* first_last_minor_upper,
            const Bool* minor_scales_with_density_lower,
            const Bool* minor_scales_with_density_upper,
            const Bool* scale_by_complement_lower,
            const Bool* scale_by_complement_upper,
            const int* idx_minor_lower,
            const int* idx_minor_upper,
            const int* idx_minor_scaling_lower,
            const int* idx_minor_scaling_upper,
            const int* kminor_start_lower,
            const int* kminor_start_upper,
            const Bool* tropo,
            const Float* col_mix, const Float* fmajor,
            const Float* fminor, const Float* play,
            const Float* tlay, const Float* col_gas,
            const int* jeta, const int* jtemp,
            const int* jpress,
            Float* tau)
    {
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid_maj(nlay, ncol_sub, 1);
        dim3 block_maj;

        if (tunings.count("gas_optical_depths_major_kernel_rt") == 0)
        {
            Float* tau_tmp = Tools_gpu::allocate_gpu<Float>(nlay*ncol_sub);

            std::tie(grid_maj, block_maj) = tune_kernel(
                    "gas_optical_depths_major_kernel_rt",
                    dim3(nlay, ncol_sub, 1),
                    {1, 2}, {64, 96, 128, 256, 512, 768, 1024}, {1},
                    gas_optical_depths_major_kernel,
                    col_s, ncol_sub, ncol, nlay, nband, ngpt,
                    nflav, neta, npres, ntemp,
                    igpt-1, band_lims_gpt,
                    kmajor, col_mix, fmajor, jeta,
                    tropo, jtemp, jpress,
                    tau_tmp);

            Tools_gpu::free_gpu<Float>(tau_tmp);

            tunings["gas_optical_depths_major_kernel_rt"].first = grid_maj;
            tunings["gas_optical_depths_major_kernel_rt"].second = block_maj;
        }
        else
        {
            block_maj = tunings["gas_optical_depths_major_kernel_rt"].second;
        }

        grid_maj = calc_grid_size(block_maj, dim3(nlay, ncol_sub, 1));

        gas_optical_depths_major_kernel<<<grid_maj, block_maj>>>(
            col_s, ncol_sub, ncol, nlay, nband, ngpt,
            nflav, neta, npres, ntemp,
            igpt-1, band_lims_gpt,
            kmajor, col_mix, fmajor, jeta,
            tropo, jtemp, jpress,
            tau);


        const int nscale_lower = nminorlower;
        const int nscale_upper = nminorupper;

        // Lower
        int idx_tropo = 1;

        dim3 grid_min_1(nlay, ncol_sub, 1), block_min_1;
        if (tunings.count("gas_optical_depths_minor_kernel_lower_rt") == 0)
        {
            Float* tau_tmp = Tools_gpu::allocate_gpu<Float>(nlay*ncol_sub);

            std::tie(grid_min_1, block_min_1) = tune_kernel(
                        "gas_optical_depths_minor_kernel_lower_rt",
                        dim3(nlay, ncol_sub, 1),
                        {1}, {32, 48, 64, 96, 128, 256, 384, 512}, {1},
                        gas_optical_depths_minor_kernel,
                        col_s, ncol_sub, ncol, nlay, ngpt, igpt-1,
                        ngas, nflav, ntemp, neta,
                        nscale_lower,
                        nminorlower,
                        nminorklower,
                        idx_h2o, idx_tropo,
                        kminor_lower,
                        minor_limits_gpt_lower,
                        first_last_minor_lower,
                        minor_scales_with_density_lower,
                        scale_by_complement_lower,
                        idx_minor_lower,
                        idx_minor_scaling_lower,
                        kminor_start_lower,
                        play, tlay, col_gas,
                        fminor, jeta, jtemp,
                        tropo,
                        tau_tmp);
            Tools_gpu::free_gpu<Float>(tau_tmp);

            tunings["gas_optical_depths_minor_kernel_lower_rt"].first = grid_min_1;
            tunings["gas_optical_depths_minor_kernel_lower_rt"].second = block_min_1;
        }
        else
        {
            block_min_1 = tunings["gas_optical_depths_minor_kernel_lower_rt"].second;
        }
        
        grid_min_1 = calc_grid_size(block_min_1, dim3(nlay, ncol_sub, 1));

        gas_optical_depths_minor_kernel<<<grid_min_1, block_min_1>>>(
                col_s, ncol_sub, ncol, nlay, ngpt, igpt-1,
                ngas, nflav, ntemp, neta,
                nscale_lower,
                nminorlower,
                nminorklower,
                idx_h2o, idx_tropo,
                kminor_lower,
                minor_limits_gpt_lower,
                first_last_minor_lower,
                minor_scales_with_density_lower,
                scale_by_complement_lower,
                idx_minor_lower,
                idx_minor_scaling_lower,
                kminor_start_lower,
                play, tlay, col_gas,
                fminor, jeta, jtemp,
                tropo, tau);

        // Upper
        idx_tropo = 0;

        dim3 grid_min_2(nlay, ncol_sub, 1), block_min_2;
        if (tunings.count("gas_optical_depths_minor_kernel_upper_rt") == 0)
        {
            Float* tau_tmp = Tools_gpu::allocate_gpu<Float>(nlay*ncol_sub);
            std::tie(grid_min_2, block_min_2) = tune_kernel(
                   "gas_optical_depths_minor_kernel_upper_rt",
                   dim3(nlay, ncol_sub, 1),
                   {1}, {32, 48, 64, 96, 128, 256, 384, 512}, {1},
                   gas_optical_depths_minor_kernel,
                   col_s, ncol_sub, ncol, nlay, ngpt, igpt-1,
                   ngas, nflav, ntemp, neta,
                   nscale_upper,
                   nminorupper,
                   nminorkupper,
                   idx_h2o, idx_tropo,
                   kminor_upper,
                   minor_limits_gpt_upper,
                   first_last_minor_upper,
                   minor_scales_with_density_upper,
                   scale_by_complement_upper,
                   idx_minor_upper,
                   idx_minor_scaling_upper,
                   kminor_start_upper,
                   play, tlay, col_gas,
                   fminor, jeta, jtemp,
                   tropo,
                   tau_tmp);
            Tools_gpu::free_gpu<Float>(tau_tmp);

            tunings["gas_optical_depths_minor_kernel_upper_rt"].first = grid_min_2;
            tunings["gas_optical_depths_minor_kernel_upper_rt"].second = block_min_2;
        }
        else
        {
            block_min_2 = tunings["gas_optical_depths_minor_kernel_upper_rt"].second;
        }
        
        grid_min_2 = calc_grid_size(block_min_2, dim3(nlay, ncol_sub, 1));

        gas_optical_depths_minor_kernel<<<grid_min_2, block_min_2>>>(
                col_s, ncol_sub, ncol, nlay, ngpt, igpt-1,
                ngas, nflav, ntemp, neta,
                nscale_upper,
                nminorupper,
                nminorkupper,
                idx_h2o, idx_tropo,
                kminor_upper,
                minor_limits_gpt_upper,
                first_last_minor_upper,
                minor_scales_with_density_upper,
                scale_by_complement_upper,
                idx_minor_upper,
                idx_minor_scaling_upper,
                kminor_start_upper,
                play, tlay, col_gas,
                fminor, jeta, jtemp,
                tropo, tau);

    }



    void Planck_source(
            const int ncol, const int nlay, const int nbnd, const int ngpt, const int igpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int nPlanckTemp,
            const Float* tlay,
            const Float* tlev,
            const Float* tsfc,
            const int sfc_lay,
            const Float* fmajor,
            const int* jeta,
            const Bool* tropo,
            const int* jtemp,
            const int* jpress,
            const int* gpoint_bands,
            const int* band_lims_gpt,
            const Float* pfracin,
            const Float temp_ref_min, const Float totplnk_delta,
            const Float* totplnk,
            Float* sfc_src,
            Float* lay_src,
            Float* lev_src_inc,
            Float* lev_src_dec,
            Float* sfc_src_jac)
    {
        Tuner_map& tunings = Tuner::get_map();

        const Float delta_Tsurf = Float(1.);

        dim3 grid(ncol, nlay, 1), block;
        if (tunings.count("Planck_source_kernel_rt") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                    "Planck_source_kernel_rt",
                    dim3(ncol, nlay, 1),
                    {16, 32, 48, 64, 96, 128, 256, 512}, {1, 2, 4, 8}, {1},
                    Planck_source_kernel,
                    ncol, nlay, nbnd, ngpt,
                    nflav, neta, npres, ntemp, nPlanckTemp, igpt-1,
                    tlay, tlev, tsfc, sfc_lay,
                    fmajor, jeta, tropo, jtemp,
                    jpress, gpoint_bands, band_lims_gpt,
                    pfracin, temp_ref_min, totplnk_delta,
                    totplnk,
                    delta_Tsurf, sfc_src, lay_src,
                    lev_src_inc, lev_src_dec,
                    sfc_src_jac);

            tunings["Planck_source_kernel_rt"].first = grid;
            tunings["Planck_source_kernel_rt"].second = block;
        }
        else
        {
            block = tunings["Planck_source_kernel_rt"].second;
        }
        
        grid = calc_grid_size(block, dim3(ncol, nlay, 1));

        Planck_source_kernel<<<grid, block>>>(
                ncol, nlay, nbnd, ngpt,
                nflav, neta, npres, ntemp, nPlanckTemp, igpt-1,
                tlay, tlev, tsfc, sfc_lay,
                fmajor, jeta, tropo, jtemp,
                jpress, gpoint_bands, band_lims_gpt,
                pfracin, temp_ref_min, totplnk_delta,
                totplnk,
                delta_Tsurf,
                sfc_src, lay_src,
                lev_src_inc, lev_src_dec,
                sfc_src_jac);
    }
}

