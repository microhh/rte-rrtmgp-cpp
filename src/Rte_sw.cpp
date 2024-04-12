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

#include "Rte_sw.h"
#include "Array.h"
#include "Optical_props.h"
#include "rrtmgp_kernels.h"


namespace rrtmgp_kernel_launcher
{
    template<typename Float>
    void apply_BC(
            int ncol, int nlay, int ngpt,
            Bool top_at_1, Array<Float,3>& gpt_flux_dn)
    {
        rrtmgp_kernels::apply_BC_0(
                &ncol, &nlay, &ngpt,
                &top_at_1, gpt_flux_dn.ptr());
    }

    template<typename Float>
    void apply_BC(
            int ncol, int nlay, int ngpt, Bool top_at_1,
            const Array<Float,2>& inc_flux, Array<Float,3>& gpt_flux_dn)
    {
        rrtmgp_kernels::apply_BC_gpt(
                &ncol, &nlay, &ngpt, &top_at_1,
                const_cast<Float*>(inc_flux.ptr()), gpt_flux_dn.ptr());
    }

    template<typename Float>
    void apply_BC(
            int ncol, int nlay, int ngpt, Bool top_at_1,
            const Array<Float,2>& inc_flux,
            const Array<Float,1>& factor,
            Array<Float,3>& gpt_flux)
    {
        rrtmgp_kernels::apply_BC_factor(
                &ncol, &nlay, &ngpt,
                &top_at_1,
                const_cast<Float*>(inc_flux.ptr()),
                const_cast<Float*>(factor.ptr()),
                gpt_flux.ptr());
    }


    void sw_solver_2stream(
            const int ncol,
            const int nlay,
            const int ngpt,
            const Bool top_at_1,
            const Array<Float,3>& tau,
            const Array<Float,3>& ssa,
            const Array<Float,3>& g,
            const Array<Float,2>& mu0,
            const Array<Float,2>& sfc_alb_dir_gpt,
            const Array<Float,2>& sfc_alb_dif_gpt,
            const Array<Float,2>& inc_flux,
            Array<Float,3>& gpt_flux_up,
            Array<Float,3>& gpt_flux_dn,
            Array<Float,3>& gpt_flux_dir,
            const Bool has_dif_bc,
            const Array<Float,2>& inc_flux_dif,
            const Bool do_broadband,
            Array<Float,3>& flux_up_loc,
            Array<Float,3>& flux_dn_loc,
            Array<Float,3>& flux_dir_loc)
    {
        rrtmgp_kernels::rte_sw_solver_2stream(
                ncol,
                nlay,
                ngpt,
                top_at_1,
                tau.ptr(),
                ssa.ptr(),
                g  .ptr(),
                mu0.ptr(),
                sfc_alb_dir_gpt.ptr(),
                sfc_alb_dif_gpt.ptr(),
                inc_flux.ptr(),
                gpt_flux_up.ptr(),
                gpt_flux_dn.ptr(),
                gpt_flux_dir.ptr(),
                has_dif_bc,
                inc_flux_dif.ptr(),
                do_broadband,
                flux_up_loc.ptr(),
                flux_dn_loc.ptr(),
                flux_dir_loc.ptr());
    }
}


void Rte_sw::rte_sw(
        const std::unique_ptr<Optical_props_arry>& optical_props,
        const Bool top_at_1,
        const Array<Float,1>& mu0,
        const Array<Float,2>& inc_flux_dir,
        const Array<Float,2>& sfc_alb_dir,
        const Array<Float,2>& sfc_alb_dif,
        const Array<Float,2>& inc_flux_dif,
        Array<Float,3>& gpt_flux_up,
        Array<Float,3>& gpt_flux_dn,
        Array<Float,3>& gpt_flux_dir)
{
    const int ncol = optical_props->get_ncol();
    const int nlay = optical_props->get_nlay();
    const int ngpt = optical_props->get_ngpt();

    Array<Float,2> sfc_alb_dir_gpt({ncol, ngpt});
    Array<Float,2> sfc_alb_dif_gpt({ncol, ngpt});

    expand_and_transpose(optical_props, sfc_alb_dir, sfc_alb_dir_gpt);
    expand_and_transpose(optical_props, sfc_alb_dif, sfc_alb_dif_gpt);


    // CvH: first, we only run with constant mu0 with height, thus we copy into height.
    Array<Float,2> mu0_2d({ncol, nlay});
    for (int j=1; j<=nlay; ++j)
        for (int i=1; i<=ncol; ++i)
            mu0_2d({i, j}) = mu0({i});

    // Run the radiative transfer solver
    // CvH: only two-stream solutions, I skipped the sw_solver_noscat
    const Bool has_dif_bc = (inc_flux_dif.size() > 0);
    const Bool do_broadband = (gpt_flux_up.dim(3) == 1) ? true : false;

    rrtmgp_kernel_launcher::sw_solver_2stream(
            ncol, nlay, ngpt, top_at_1,
            optical_props->get_tau(),
            optical_props->get_ssa(),
            optical_props->get_g  (),
            mu0_2d,
            sfc_alb_dir_gpt, sfc_alb_dif_gpt,
            inc_flux_dir,
            gpt_flux_up, gpt_flux_dn, gpt_flux_dir,
            has_dif_bc, inc_flux_dif,
            do_broadband, gpt_flux_up, gpt_flux_dn, gpt_flux_dir);

    // CvH: The original fortran code had a call to the reduce here.
    // fluxes->reduce(gpt_flux_up, gpt_flux_dn, gpt_flux_dir, optical_props, top_at_1);
}


void Rte_sw::expand_and_transpose(
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
