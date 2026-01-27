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

#include "Rte_lw_rt.h"
#include "Array.h"
#include "Optical_props_rt.h"
#include "Source_functions_rt.h"
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
                const int idx_in  = ibnd + icol*nbnd;
                const int idx_out = icol + igpt*ncol;

                arr_out[idx_out] = arr_in[idx_in];
            }
        }
    }
}

//    
//    void apply_BC(
//            int ncol, int nlay, int ngpt,
//            Bool top_at_1, Array<Float,3>& gpt_flux_dn)
//    {
//        rrtmgp_kernels::apply_BC_0(
//                &ncol, &nlay, &ngpt,
//                &top_at_1, gpt_flux_dn.ptr());
//    }
//
//    
//    void apply_BC(
//            int ncol, int nlay, int ngpt,
//            Bool top_at_1, const Array<Float,2>& inc_flux,
//            Array<Float,3>& gpt_flux_dn)
//    {
//        rrtmgp_kernels::apply_BC_gpt(
//                &ncol, &nlay, &ngpt,
//                &top_at_1, const_cast<Float*>(inc_flux.ptr()), gpt_flux_dn.ptr());
//    }


void Rte_lw_rt::rte_lw(
        const std::unique_ptr<Optical_props_arry_rt>& optical_props,
        const Bool top_at_1,
        const Source_func_lw_rt& sources,
        const Array_gpu<Float,2>& sfc_emis,
        const Array_gpu<Float,1>& inc_flux,
        Array_gpu<Float,2>& gpt_flux_up,
        Array_gpu<Float,2>& gpt_flux_dn,
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

    // Upper boundary condition.
    if (inc_flux.size() == 0)
        Rte_solver_kernels_cuda_rt::apply_BC(ncol, nlay, top_at_1, gpt_flux_dn.ptr());
    else
        Rte_solver_kernels_cuda_rt::apply_BC(ncol, nlay, top_at_1, inc_flux.ptr(), gpt_flux_dn.ptr());

    // Run the radiative transfer solver.
    const int n_quad_angs = n_gauss_angles;

    Array_gpu<Float,2> gauss_Ds_subset = gauss_Ds.subset(
            {{ {1, n_quad_angs}, {n_quad_angs, n_quad_angs} }});
    Array_gpu<Float,2> gauss_wts_subset = gauss_wts.subset(
            {{ {1, n_quad_angs}, {n_quad_angs, n_quad_angs} }});

    // For now, just pass the arrays around.
    Array_gpu<Float,1> sfc_src_jac(sources.get_sfc_source().get_dims());
    Array_gpu<Float,2> gpt_flux_up_jac(gpt_flux_up.get_dims());

    Rte_solver_kernels_cuda_rt::lw_solver_noscat_gaussquad(
            ncol, nlay, top_at_1, n_quad_angs,
            gauss_Ds_subset.ptr(), gauss_wts_subset.ptr(),
            optical_props->get_tau().ptr(),
            sources.get_lay_source().ptr(),
            sources.get_lev_source().ptr(),
            sfc_emis.ptr(), sources.get_sfc_source().ptr(),
            gpt_flux_up.ptr(), gpt_flux_dn.ptr(),
            sfc_src_jac.ptr(), gpt_flux_up_jac.ptr());

    // CvH: In the fortran code this call is here, I removed it for performance and flexibility.
    // fluxes->reduce(gpt_flux_up, gpt_flux_dn, optical_props, top_at_1);
}


void Rte_lw_rt::expand_and_transpose(
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

    expand_and_transpose_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nbnd, limits.ptr(), arr_out.ptr(), arr_in.ptr());
}
