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

#include "rte_lw.h"
#include "array.h"
#include "optical_props.h"
#include "source_functions.h"

// #include "rrtmgp_kernels.h"
#include "rte_solver_kernels_cuda.h"


namespace
{
    template<typename Float>__global__
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
                const int idx_in  = ibnd + icol*nbnd;
                const int idx_out = icol + igpt*ncol;

                arr_out[idx_out] = arr_in[idx_in];
            }
        }
    }
}


void Rte_lw_gpu::rte_lw(
        const std::unique_ptr<Optical_props_arry_gpu>& optical_props,
        const bool top_at_1,
        const Source_func_lw_gpu& sources,
        const Array_gpu<Float,2>& sfc_emis,
        const Array_gpu<Float,2>& inc_flux,
        Array_gpu<Float,3>& gpt_flux_up,
        Array_gpu<Float,3>& gpt_flux_dn,
        const int n_gauss_angles)
{
    const int max_gauss_pts = 4;

    // Weights and angle secants for "Gauss-Jacobi-5" quadrature.
    // Values from Table 1, R. J. Hogan 2023, doi:10.1002/qj.4598
    const Array_gpu<Float,2> gauss_Ds(
            Array<Float,2>(
            { 1./0.6096748751, 0.            , 0.             , 0.,
              1./0.2509907356, 1/0.7908473988, 0.             , 0.,
              1./0.1024922169, 1/0.4417960320, 1./0.8633751621, 0.,
              1./0.0454586727, 1/0.2322334416, 1./0.5740198775, 1./0.903077597 },
            { max_gauss_pts, max_gauss_pts }));

    const Array_gpu<Float,2> gauss_wts(
            Array<Float,2>(
            { 1.,           0.,           0.,           0.,
              0.2300253764, 0.7699746236, 0.,           0.,
              0.0437820218, 0.3875796738, 0.5686383044, 0.,
              0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087 },
            { max_gauss_pts, max_gauss_pts }));

    const int ncol = optical_props->get_ncol();
    const int nlay = optical_props->get_nlay();
    const int ngpt = optical_props->get_ngpt();

    Array_gpu<Float,2> sfc_emis_gpt({ncol, ngpt});
    expand_and_transpose(optical_props, sfc_emis, sfc_emis_gpt);

    // Run the radiative transfer solver.
    const int n_quad_angs = n_gauss_angles;

    Array_gpu<Float,2> gauss_wts_subset = gauss_wts.subset(
            {{ {1, n_quad_angs}, {n_quad_angs, n_quad_angs} }});

    Array_gpu<Float,3> secants({ncol, ngpt, n_quad_angs});
    Rte_solver_kernels_cuda::lw_secants_array(
            ncol, ngpt, n_quad_angs, max_gauss_pts,
            gauss_Ds.ptr(), secants.ptr());

    // For now, just pass the arrays around.
    Array_gpu<Float,2> sfc_src_jac(sources.get_sfc_source().get_dims());
    Array_gpu<Float,3> gpt_flux_up_jac(gpt_flux_up.get_dims());

    const bool do_broadband = (gpt_flux_up.dim(3) == 1) ? true : false;

    if (do_broadband)
        throw std::runtime_error("Broadband fluxes not implemented, performance gain on GPU is negligible");

    const bool do_jacobians = false;

    // pass null ptr if size of inc_flux is zero
    const Float* inc_flux_ptr = (inc_flux.size() == 0) ? nullptr : inc_flux.ptr();

    Rte_solver_kernels_cuda::lw_solver_noscat(
            ncol, nlay, ngpt, top_at_1, n_quad_angs,
            secants.ptr(), gauss_wts_subset.ptr(),
            optical_props->get_tau().ptr(),
            sources.get_lay_source().ptr(),
            sources.get_lev_source().ptr(),
            sfc_emis_gpt.ptr(), sources.get_sfc_source().ptr(),
            inc_flux_ptr,
            gpt_flux_up.ptr(), gpt_flux_dn.ptr(),
            do_broadband, gpt_flux_up.ptr(), gpt_flux_dn.ptr(),
            do_jacobians, sfc_src_jac.ptr(), gpt_flux_up_jac.ptr());

    // CvH: In the fortran code this call is here, I removed it for performance and flexibility.
    // fluxes->reduce(gpt_flux_up, gpt_flux_dn, optical_props, top_at_1);
}


void Rte_lw_gpu::expand_and_transpose(
        const std::unique_ptr<Optical_props_arry_gpu>& ops,
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

    expand_and_transpose_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nbnd, limits.ptr(), arr_out.ptr(), arr_in.ptr());
}
