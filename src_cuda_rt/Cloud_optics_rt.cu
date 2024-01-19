/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/earth-system-radiation/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/earth-system-radiation/rte-rrtmgp-cpp
 *
 * Contact: Chiel van Heerwaarden
 * email: chiel.vanheerwaarden@wur.nl
 *
 * Copyright 2020, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#include "Cloud_optics_rt.h"

namespace
{
    __global__
    void compute_from_table_kernel(
            const int ncol, const int nlay, const int ibnd, const Bool* mask,
            const Float* cwp, const Float* re,
            const int nsteps, const Float step_size, const Float offset,
            const Float* tau_table, const Float* ssa_table, const Float* asy_table,
            Float* tau, Float* taussa, Float* taussag)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( icol < ncol) && ( ilay < nlay) )
        {
            const int idx = icol + ilay*ncol;

            if (mask[idx])
            {
                const Float reff= min(Float(21.5), re[idx]);
                const int index = min(int((reff - offset) / step_size) + 1, nsteps-1) - 1;

                const int idx_ib = index + ibnd*nsteps;
                const Float fint = (reff - offset) /step_size - (index);

                const Float tau_local = cwp[idx] *
                                     (tau_table[idx_ib] + fint * (tau_table[idx_ib+1] - tau_table[idx_ib]));
                const Float taussa_local = tau_local *
                                     (ssa_table[idx_ib] + fint * (ssa_table[idx_ib+1] - ssa_table[idx_ib]));
                const Float taussag_local = taussa_local *
                                     (asy_table[idx_ib] + fint * (asy_table[idx_ib+1] - asy_table[idx_ib]));

                tau[idx]     = tau_local;
                taussa[idx]  = taussa_local;
                taussag[idx] = taussag_local;
            }
            else
            {
                tau[idx]     = Float(0.);
                taussa[idx]  = Float(0.);
                taussag[idx] = Float(0.);
            }
        }
    }

    __global__
    void combine_and_store_kernel(const int ncol, const int nlay, const Float tmin,
                  Float* __restrict__ tau,
                  const Float* __restrict__ ltau, const Float* __restrict__ ltaussa,
                  const Float* __restrict__ itau, const Float* __restrict__ itaussa)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (ilay < nlay) )
        {
            const int idx = icol + ilay*ncol;
            const Float tau_t = (ltau[idx] - ltaussa[idx]) + (itau[idx] - itaussa[idx]);

            tau[idx] = tau_t;
        }
    }

    __global__
    void combine_and_store_kernel(const int ncol, const int nlay, const Float tmin,
                  Float* __restrict__ tau, Float* __restrict__ ssa, Float* __restrict__ g,
                  const Float* __restrict__ ltau, const Float* __restrict__ ltaussa, const Float* __restrict__ ltaussag,
                  const Float* __restrict__ itau, const Float* __restrict__ itaussa, const Float* __restrict__ itaussag)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (ilay < nlay) )
        {
            const int idx = icol + ilay*ncol;
            const Float tau_t = ltau[idx] + itau[idx];
            const Float taussa = ltaussa[idx] + itaussa[idx];
            const Float taussag = ltaussag[idx] + itaussag[idx];

            tau[idx] = tau_t;
            ssa[idx] = taussa / max(tau_t, tmin);
            g[idx]   = taussag/ max(taussa, tmin);
        }
    }

    __global__
    void set_mask(const int ncol, const int nlay, const Float min_value,
                  Bool* __restrict__ mask, const Float* __restrict__ values)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (ilay < nlay) )
        {
            const int idx = icol + ilay*ncol;
            mask[idx] = values[idx] > min_value;
        }
    }
}



Cloud_optics_rt::Cloud_optics_rt(
        const Array<Float,2>& band_lims_wvn,
        const Float radliq_lwr, const Float radliq_upr, const Float radliq_fac,
        const Float radice_lwr, const Float radice_upr, const Float radice_fac,
        const Array<Float,2>& lut_extliq, const Array<Float,2>& lut_ssaliq, const Array<Float,2>& lut_asyliq,
        const Array<Float,3>& lut_extice, const Array<Float,3>& lut_ssaice, const Array<Float,3>& lut_asyice) :
    Optical_props_rt(band_lims_wvn)
{
    const int nsize_liq = lut_extliq.dim(1);
    const int nsize_ice = lut_extice.dim(1);

    this->liq_nsteps = nsize_liq;
    this->ice_nsteps = nsize_ice;
    this->liq_step_size = (radliq_upr - radliq_lwr) / (nsize_liq - Float(1.));
    this->ice_step_size = (radice_upr - radice_lwr) / (nsize_ice - Float(1.));

    // Load LUT constants.
    this->radliq_lwr = radliq_lwr;
    this->radliq_upr = radliq_upr;
    this->radice_lwr = radice_lwr;
    this->radice_upr = radice_upr;

    // Load LUT coefficients.
    this->lut_extliq = lut_extliq;
    this->lut_ssaliq = lut_ssaliq;
    this->lut_asyliq = lut_asyliq;

    // Choose the intermediately rough ice particle category (icergh = 2).
    this->lut_extice.set_dims({lut_extice.dim(1), lut_extice.dim(2)});
    this->lut_ssaice.set_dims({lut_ssaice.dim(1), lut_ssaice.dim(2)});
    this->lut_asyice.set_dims({lut_asyice.dim(1), lut_asyice.dim(2)});

    constexpr int icergh = 2;
    for (int ibnd=1; ibnd<=lut_extice.dim(2); ++ibnd)
        for (int isize=1; isize<=lut_extice.dim(1); ++isize)
        {
            this->lut_extice({isize, ibnd}) = lut_extice({isize, ibnd, icergh});
            this->lut_ssaice({isize, ibnd}) = lut_ssaice({isize, ibnd, icergh});
            this->lut_asyice({isize, ibnd}) = lut_asyice({isize, ibnd, icergh});
        }

    this->lut_extice_gpu = this->lut_extice;
    this->lut_ssaice_gpu = this->lut_ssaice;
    this->lut_asyice_gpu = this->lut_asyice;
    this->lut_extliq_gpu = this->lut_extliq;
    this->lut_ssaliq_gpu = this->lut_ssaliq;
    this->lut_asyliq_gpu = this->lut_asyliq;
}


// Two-stream variant of cloud optics.
void Cloud_optics_rt::cloud_optics(
        const int ibnd,
        const Array_gpu<Float,2>& clwp, const Array_gpu<Float,2>& ciwp,
        const Array_gpu<Float,2>& reliq, const Array_gpu<Float,2>& reice,
        Optical_props_2str_rt& optical_props)
{
    const int ncol = clwp.dim(1);
    const int nlay = clwp.dim(2);

    Optical_props_2str_rt clouds_liq(ncol, nlay, optical_props);
    Optical_props_2str_rt clouds_ice(ncol, nlay, optical_props);

    // Set the mask.
    constexpr Float mask_min_value = Float(0.);
    const int block_col_m = 16;
    const int block_lay_m = 16;

    const int grid_col_m  = ncol/block_col_m + (ncol%block_col_m > 0);
    const int grid_lay_m  = nlay/block_lay_m + (nlay%block_lay_m > 0);

    dim3 grid_m_gpu(grid_col_m, grid_lay_m);
    dim3 block_m_gpu(block_col_m, block_lay_m);

    Array_gpu<Bool,2> liqmsk({ncol, nlay});
    set_mask<<<grid_m_gpu, block_m_gpu>>>(
            ncol, nlay, mask_min_value, liqmsk.ptr(), clwp.ptr());

    Array_gpu<Bool,2> icemsk({ncol, nlay});
    set_mask<<<grid_m_gpu, block_m_gpu>>>(
            ncol, nlay, mask_min_value, icemsk.ptr(), ciwp.ptr());

    // Temporary arrays for storage.
    Array_gpu<Float,2> ltau    ({ncol, nlay});
    Array_gpu<Float,2> ltaussa ({ncol, nlay});
    Array_gpu<Float,2> ltaussag({ncol, nlay});

    Array_gpu<Float,2> itau    ({ncol, nlay});
    Array_gpu<Float,2> itaussa ({ncol, nlay});
    Array_gpu<Float,2> itaussag({ncol, nlay});

    const int block_col = 64;
    const int block_lay = 1;

    const int grid_col  = ncol/block_col + (ncol%block_col > 0);
    const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

    dim3 grid_gpu(grid_col, grid_lay);
    dim3 block_gpu(block_col, block_lay);

    // Liquid water
    compute_from_table_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, ibnd-1, liqmsk.ptr(), clwp.ptr(), reliq.ptr(),
            this->liq_nsteps, this->liq_step_size, this->radliq_lwr,
            this->lut_extliq_gpu.ptr(), this->lut_ssaliq_gpu.ptr(),
            this->lut_asyliq_gpu.ptr(), ltau.ptr(), ltaussa.ptr(), ltaussag.ptr());

    // Ice.
    compute_from_table_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, ibnd-1, icemsk.ptr(), ciwp.ptr(), reice.ptr(),
            this->ice_nsteps, this->ice_step_size, this->radice_lwr,
            this->lut_extice_gpu.ptr(), this->lut_ssaice_gpu.ptr(),
            this->lut_asyice_gpu.ptr(), itau.ptr(), itaussa.ptr(), itaussag.ptr());

    constexpr Float eps = std::numeric_limits<Float>::epsilon();

    combine_and_store_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, eps,
            optical_props.get_tau().ptr(), optical_props.get_ssa().ptr(), optical_props.get_g().ptr(),
            ltau.ptr(), ltaussa.ptr(), ltaussag.ptr(),
            itau.ptr(), itaussa.ptr(), itaussag.ptr());
}

// 1scl variant of cloud optics.
void Cloud_optics_rt::cloud_optics(
        const int ibnd,
        const Array_gpu<Float,2>& clwp, const Array_gpu<Float,2>& ciwp,
        const Array_gpu<Float,2>& reliq, const Array_gpu<Float,2>& reice,
        Optical_props_1scl_rt& optical_props)
{
    const int ncol = clwp.dim(1);
    const int nlay = clwp.dim(2);

    Optical_props_1scl_rt clouds_liq(ncol, nlay, optical_props);
    Optical_props_1scl_rt clouds_ice(ncol, nlay, optical_props);

    // Set the mask.
    constexpr Float mask_min_value = Float(0.);
    const int block_col_m = 16;
    const int block_lay_m = 16;

    const int grid_col_m  = ncol/block_col_m + (ncol%block_col_m > 0);
    const int grid_lay_m  = nlay/block_lay_m + (nlay%block_lay_m > 0);

    dim3 grid_m_gpu(grid_col_m, grid_lay_m);
    dim3 block_m_gpu(block_col_m, block_lay_m);

    Array_gpu<Bool,2> liqmsk({ncol, nlay});
    set_mask<<<grid_m_gpu, block_m_gpu>>>(
            ncol, nlay, mask_min_value, liqmsk.ptr(), clwp.ptr());

    Array_gpu<Bool,2> icemsk({ncol, nlay});
    set_mask<<<grid_m_gpu, block_m_gpu>>>(
            ncol, nlay, mask_min_value, icemsk.ptr(), ciwp.ptr());

    // Temporary arrays for storage.
    Array_gpu<Float,2> ltau    ({ncol, nlay});
    Array_gpu<Float,2> ltaussa ({ncol, nlay});
    Array_gpu<Float,2> ltaussag({ncol, nlay});

    Array_gpu<Float,2> itau    ({ncol, nlay});
    Array_gpu<Float,2> itaussa ({ncol, nlay});
    Array_gpu<Float,2> itaussag({ncol, nlay});

    const int block_col = 64;
    const int block_lay = 1;

    const int grid_col  = ncol/block_col + (ncol%block_col > 0);
    const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

    dim3 grid_gpu(grid_col, grid_lay);
    dim3 block_gpu(block_col, block_lay);

    // Liquid water
    compute_from_table_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, ibnd-1, liqmsk.ptr(), clwp.ptr(), reliq.ptr(),
            this->liq_nsteps, this->liq_step_size, this->radliq_lwr,
            this->lut_extliq_gpu.ptr(), this->lut_ssaliq_gpu.ptr(),
            this->lut_asyliq_gpu.ptr(), ltau.ptr(), ltaussa.ptr(), ltaussag.ptr());

    // Ice.
    compute_from_table_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, ibnd-1, icemsk.ptr(), ciwp.ptr(), reice.ptr(),
            this->ice_nsteps, this->ice_step_size, this->radice_lwr,
            this->lut_extice_gpu.ptr(), this->lut_ssaice_gpu.ptr(),
            this->lut_asyice_gpu.ptr(), itau.ptr(), itaussa.ptr(), itaussag.ptr());

    constexpr Float eps = std::numeric_limits<Float>::epsilon();

    combine_and_store_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, eps,
            optical_props.get_tau().ptr(),
            ltau.ptr(), ltaussa.ptr(),
            itau.ptr(), itaussa.ptr());
}

