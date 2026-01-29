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

#include "Rte_sw_rt.h"
#include "Array.h"
#include "Optical_props_rt.h"
#include "Fluxes_rt.h"

// #include "rrtmgp_kernels.h"
#include "rte_solver_kernels_cuda_rt.h"


namespace
{
    __global__
    void expand_and_transpose_kernel(
        const int ncol, const int nbnd, const int* __restrict__ limits,
        Float* __restrict__ arr_out, const Float* __restrict__ arr_in)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ibnd = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( icol < ncol) && (ibnd < nbnd) )
        {
            const int gpt_start = limits[2*ibnd] - 1;
            const int gpt_end = limits[2*ibnd+1];

            for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
            {
                const int idx_in = ibnd + icol*nbnd;
                const int idx_out = icol + igpt*ncol;

                arr_out[idx_out] = arr_in[idx_in];
            }
        }
    }
}

//namespace rrtmgp_kernel_launcher
//{
//
//    void apply_BC(
//            int ncol, int nlay, int ngpt,
//            bool top_at_1, Array<Float,3>& gpt_flux_dn)
//    {
//        rrtmgp_kernels::apply_BC_0(
//                &ncol, &nlay, &ngpt,
//                &top_at_1, gpt_flux_dn.ptr());
//    }
//
//
//    void apply_BC(
//            int ncol, int nlay, int ngpt, bool top_at_1,
//            const Array<Float,2>& inc_flux, Array<Float,3>& gpt_flux_dn)
//    {
//        rrtmgp_kernels::apply_BC_gpt(
//                &ncol, &nlay, &ngpt, &top_at_1,
//                const_cast<Float*>(inc_flux.ptr()), gpt_flux_dn.ptr());
//    }
//
//
//    void apply_BC(
//            int ncol, int nlay, int ngpt, bool top_at_1,
//            const Array<Float,2>& inc_flux,
//            const Array<Float,1>& factor,
//            Array<Float,3>& gpt_flux)
//    {
//        rrtmgp_kernels::apply_BC_factor(
//                &ncol, &nlay, &ngpt,
//                &top_at_1,
//                const_cast<Float*>(inc_flux.ptr()),
//                const_cast<Float*>(factor.ptr()),
//                gpt_flux.ptr());
//    }
//
//
//    void sw_solver_2stream(
//            int ncol, int nlay, int ngpt, bool top_at_1,
//            const Array<Float,3>& tau,
//            const Array<Float,3>& ssa,
//            const Array<Float,3>& g,
//            const Array<Float,1>& mu0,
//            const Array<Float,2>& sfc_alb_dir_gpt, const Array<Float,2>& sfc_alb_dif_gpt,
//            Array<Float,3>& gpt_flux_up, Array<Float,3>& gpt_flux_dn, Array<Float,3>& gpt_flux_dir)
//    {
//        rrtmgp_kernels::sw_solver_2stream(
//                &ncol, &nlay, &ngpt, &top_at_1,
//                const_cast<Float*>(tau.ptr()),
//                const_cast<Float*>(ssa.ptr()),
//                const_cast<Float*>(g  .ptr()),
//                const_cast<Float*>(mu0.ptr()),
//                const_cast<Float*>(sfc_alb_dir_gpt.ptr()),
//                const_cast<Float*>(sfc_alb_dif_gpt.ptr()),
//                gpt_flux_up.ptr(), gpt_flux_dn.ptr(), gpt_flux_dir.ptr());
//    }


void Rte_sw_rt::rte_sw(
        const std::unique_ptr<Optical_props_arry_rt>& optical_props,
        const bool top_at_1,
        const Array_gpu<Float,1>& mu0,
        const Array_gpu<Float,1>& inc_flux_dir,
        const Array_gpu<Float,2>& sfc_alb_dir,
        const Array_gpu<Float,2>& sfc_alb_dif,
        const Array_gpu<Float,1>& inc_flux_dif,
        Array_gpu<Float,2>& gpt_flux_up,
        Array_gpu<Float,2>& gpt_flux_dn,
        Array_gpu<Float,2>& gpt_flux_dir)
{
    const int ncol = optical_props->get_ncol();
    const int nlay = optical_props->get_nlay();

 //   expand_and_transpose(optical_props, sfc_alb_dir, sfc_alb_dir_gpt);
 //   expand_and_transpose(optical_props, sfc_alb_dif, sfc_alb_dif_gpt);

    // Upper boundary condition. At this stage, flux_dn contains the diffuse radiation only.
    Rte_solver_kernels_cuda_rt::apply_BC(ncol, nlay, top_at_1, inc_flux_dir.ptr(), mu0.ptr(), gpt_flux_dir.ptr());
    if (inc_flux_dif.size() == 0)
        Rte_solver_kernels_cuda_rt::apply_BC(ncol, nlay, top_at_1, gpt_flux_dn.ptr());
    else
        Rte_solver_kernels_cuda_rt::apply_BC(ncol, nlay, top_at_1, inc_flux_dif.ptr(), gpt_flux_dn.ptr());

    // Run the radiative transfer solver
    // CvH: only two-stream solutions, I skipped the sw_solver_noscat.
    Rte_solver_kernels_cuda_rt::sw_solver_2stream(
            ncol, nlay, top_at_1,
            optical_props->get_tau().ptr(),
            optical_props->get_ssa().ptr(),
            optical_props->get_g  ().ptr(),
            mu0.ptr(),
            sfc_alb_dir.ptr(), sfc_alb_dif.ptr(),
            gpt_flux_up.ptr(), gpt_flux_dn.ptr(), gpt_flux_dir.ptr());

    // CvH: The original fortran code had a call to the reduce here.
    // fluxes->reduce(gpt_flux_up, gpt_flux_dn, gpt_flux_dir, optical_props, top_at_1);
}


void Rte_sw_rt::expand_and_transpose(
        const std::unique_ptr<Optical_props_arry_rt>& ops,
        const Array_gpu<Float,2> arr_in,
        Array_gpu<Float,2>& arr_out)
{
    const int ncol = arr_in.dim(2);
    const int nbnd = ops->get_nband();
    const int block_col = 16;
    const int block_bnd = 14;

    const int grid_col = ncol/block_col + (ncol%block_col > 0);
    const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

    dim3 grid_gpu(grid_col, grid_bnd);
    dim3 block_gpu(block_col, block_bnd);

    Array_gpu<int,2> limits = ops->get_band_lims_gpoint_gpu();

    // Array_gpu<int,2> limits(limitsc);
    expand_and_transpose_kernel<<<grid_gpu, block_gpu>>>(
        ncol, nbnd, limits.ptr(), arr_out.ptr(), arr_in.ptr());
}

