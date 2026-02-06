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
#include "rrtmgp_kernels.h"


namespace rrtmgp_kernel_launcher
{
    /*
    void apply_BC(
            const int ncol,
            const int nlay,
            const int ngpt,
            const Bool top_at_1,
            Array<Float,3>& gpt_flux_dn)
    {
        rrtmgp_kernels::apply_BC_0(
                ncol,
                nlay,
                ngpt,
                top_at_1,
                gpt_flux_dn.ptr());
    }


    void apply_BC(
            const int ncol,
            const int nlay,
            const int ngpt,
            const Bool top_at_1,
            const Array<Float,2>& inc_flux,
            Array<Float,3>& gpt_flux_dn)
    {
        rrtmgp_kernels::apply_BC_gpt(
                ncol,
                nlay,
                ngpt,
                top_at_1,
                inc_flux.ptr(),
                gpt_flux_dn.ptr());
    }
    */


    template<typename Float>
    void lw_solver_noscat_GaussQuad(
            const int ncol,
            const int nlay,
            const int ngpt,
            const Bool top_at_1,
            const int n_quad_angs,
            const Array<Float,3>& secants,
            const Array<Float,2>& gauss_wts_subset,
            const Array<Float,3>& tau,
            const Array<Float,3>& lay_source,
            const Array<Float,3>& lev_source,
            const Array<Float,2>& sfc_emis_gpt,
            const Array<Float,2>& sfc_source,
            const Array<Float,2>& inc_flux_diffuse,
            Array<Float,3>& gpt_flux_up,
            Array<Float,3>& gpt_flux_dn,
            const Bool do_broadband,
            Array<Float,3>& flux_up_loc,
            Array<Float,3>& flux_dn_loc,
            const Bool do_jacobians,
            const Array<Float,2>& sfc_source_jac,
            Array<Float,3>& gpt_flux_up_jac,
            const Bool do_rescaling,
            const Array<Float,3>& ssa,
            const Array<Float,3>& g)
    {
        rrtmgp_kernels::rte_lw_solver_noscat(
                ncol,
                nlay,
                ngpt,
                top_at_1,
                n_quad_angs,
                secants.ptr(),
                gauss_wts_subset.ptr(),
                tau.ptr(),
                lay_source.ptr(),
                lev_source.ptr(),
                sfc_emis_gpt.ptr(),
                sfc_source.ptr(),
                inc_flux_diffuse.ptr(),
                gpt_flux_up.ptr(),
                gpt_flux_dn.ptr(),
                do_broadband,
                flux_up_loc.ptr(),
                flux_dn_loc.ptr(),
                do_jacobians,
                sfc_source_jac.ptr(),
                gpt_flux_up_jac.ptr(),
                do_rescaling,
                ssa.ptr(),
                g.ptr());
    }
}


void Rte_lw::rte_lw(
        const std::unique_ptr<Optical_props_arry>& optical_props,
        const Bool top_at_1,
        const Source_func_lw& sources,
        const Array<Float,2>& sfc_emis,
        const Array<Float,2>& inc_flux,
        Array<Float,3>& gpt_flux_up,
        Array<Float,3>& gpt_flux_dn,
        const int n_gauss_angles)
{
    const int max_gauss_pts = 4;

    // Weights and angle secants for "Gauss-Jacobi-5" quadrature.
    // Values from Table 1, R. J. Hogan 2023, doi:10.1002/qj.4598
    const Array<Float,2> gauss_Ds(
            { 1./0.6096748751, 0.            , 0.             , 0.,
              1./0.2509907356, 1/0.7908473988, 0.             , 0.,
              1./0.1024922169, 1/0.4417960320, 1./0.8633751621, 0.,
              1./0.0454586727, 1/0.2322334416, 1./0.5740198775, 1./0.903077597 },
            { max_gauss_pts, max_gauss_pts });

    const Array<Float,2> gauss_wts(
            { 1.,           0.,           0.,           0.,
              0.2300253764, 0.7699746236, 0.,           0.,
              0.0437820218, 0.3875796738, 0.5686383044, 0.,
              0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087 },
            { max_gauss_pts, max_gauss_pts });

    const int ncol = optical_props->get_ncol();
    const int nlay = optical_props->get_nlay();
    const int ngpt = optical_props->get_ngpt();

    Array<Float,2> sfc_emis_gpt({ncol, ngpt});

    expand_and_transpose(optical_props, sfc_emis, sfc_emis_gpt);

    // Run the radiative transfer solver
    const int n_quad_angs = n_gauss_angles;

    // Array<Float,2> gauss_Ds_subset = gauss_Ds.subset(
    //        {{ {1, n_quad_angs}, {n_quad_angs, n_quad_angs} }});
    Array<Float,2> gauss_wts_subset = gauss_wts.subset(
            {{ {1, n_quad_angs}, {n_quad_angs, n_quad_angs} }});

    Array<Float,3> secants({ncol, ngpt, n_quad_angs});
    for (int imu=1; imu<=n_quad_angs; ++imu)
        for (int igpt=1; igpt<=ngpt; ++igpt)
            for (int icol=1; icol<=ncol; ++icol)
                secants({icol, igpt, imu}) = gauss_Ds({imu, n_quad_angs});

    const Bool do_broadband = (gpt_flux_up.dim(3) == 1) ? true : false;

    // CvH: For now, just pass the arrays around. Could we reduce the array size?
    const Bool do_jacobians = false;
    Array<Float,2> sfc_src_jac(sources.get_sfc_source().get_dims());
    Array<Float,3> gpt_flux_up_jac(gpt_flux_up.get_dims());

    // Do not rescale and pass tau in twice in the last line to not trigger an exception.
    const Bool do_rescaling = false;

    rrtmgp_kernel_launcher::lw_solver_noscat_GaussQuad(
            ncol, nlay, ngpt, top_at_1, n_quad_angs,
            secants, gauss_wts_subset,
            optical_props->get_tau(),
            sources.get_lay_source(),
            sources.get_lev_source(),
            sfc_emis_gpt, sources.get_sfc_source(),
            inc_flux,
            gpt_flux_up, gpt_flux_dn,
            do_broadband, gpt_flux_up, gpt_flux_dn,
            do_jacobians, sfc_src_jac, gpt_flux_up_jac,
            do_rescaling, optical_props->get_tau(), optical_props->get_tau());

    // CvH: In the fortran code this call is here, I removed it for performance and flexibility.
    // fluxes->reduce(gpt_flux_up, gpt_flux_dn, optical_props, top_at_1);
}


void Rte_lw::expand_and_transpose(
        const std::unique_ptr<Optical_props_arry>& ops,
        const Array<Float,2> arr_in,
        Array<Float,2>& arr_out)
{
    const int ncol = arr_in.dim(2);
    const int nband = ops->get_nband();
    Array<int,2> limits = ops->get_band_lims_gpoint();

    for (int iband=1; iband<=nband; ++iband)
        for (int icol=1; icol<=ncol; ++icol)
            for (int igpt=limits({1, iband}); igpt<=limits({2, iband}); ++igpt)
                arr_out({icol, igpt}) = arr_in({iband, icol});
}
