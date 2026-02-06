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

#include "optical_props.h"
#include "array.h"
#include "rrtmgp_kernels.h"


// Optical properties per gpoint.
Optical_props::Optical_props(
        const Array<Float,2>& band_lims_wvn,
        const Array<int,2>& band_lims_gpt)
{
    Array<int,2> band_lims_gpt_lcl(band_lims_gpt);

    this->band2gpt = band_lims_gpt_lcl;
    this->band_lims_wvn = band_lims_wvn;

    // Make a map between g-points and bands.
    this->gpt2band.set_dims({band_lims_gpt_lcl.max()});
    for (int iband=1; iband<=band_lims_gpt_lcl.dim(2); ++iband)
    {
        for (int i=band_lims_gpt_lcl({1,iband}); i<=band_lims_gpt_lcl({2,iband}); ++i)
            this->gpt2band({i}) = iband;
    }
}


// Optical properties per band.
Optical_props::Optical_props(
        const Array<Float,2>& band_lims_wvn)
{
    Array<int,2> band_lims_gpt_lcl({2, band_lims_wvn.dim(2)});

    for (int iband=1; iband<=band_lims_wvn.dim(2); ++iband)
    {
        band_lims_gpt_lcl({1, iband}) = iband;
        band_lims_gpt_lcl({2, iband}) = iband;
    }

    this->band2gpt = band_lims_gpt_lcl;
    this->band_lims_wvn = band_lims_wvn;

    // Make a map between g-points and bands.
    this->gpt2band.set_dims({band_lims_gpt_lcl.max()});
    for (int iband=1; iband<=band_lims_gpt_lcl.dim(2); ++iband)
    {
        for (int i=band_lims_gpt_lcl({1,iband}); i<=band_lims_gpt_lcl({2,iband}); ++i)
            this->gpt2band({i}) = iband;
    }
}


Optical_props_1scl::Optical_props_1scl(
        const int ncol,
        const int nlay,
        const Optical_props& optical_props) :
    Optical_props_arry(optical_props),
    tau({ncol, nlay, this->get_ngpt()})
{}


void Optical_props_1scl::set_subset(
        const std::unique_ptr<Optical_props_arry>& optical_props_sub,
        const int col_s, const int col_e)
{
    for (int igpt=1; igpt<=tau.dim(3); ++igpt)
        for (int ilay=1; ilay<=tau.dim(2); ++ilay)
            for (int icol=col_s; icol<=col_e; ++icol)
                tau({icol, ilay, igpt}) = optical_props_sub->get_tau()({icol-col_s+1, ilay, igpt});
}


void Optical_props_1scl::get_subset(
        const std::unique_ptr<Optical_props_arry>& optical_props_sub,
        const int col_s, const int col_e)
{
    for (int igpt=1; igpt<=tau.dim(3); ++igpt)
        for (int ilay=1; ilay<=tau.dim(2); ++ilay)
            for (int icol=col_s; icol<=col_e; ++icol)
                tau({icol-col_s+1, ilay, igpt}) = optical_props_sub->get_tau()({icol, ilay, igpt});
}


Optical_props_2str::Optical_props_2str(
        const int ncol,
        const int nlay,
        const Optical_props& optical_props) :
    Optical_props_arry(optical_props),
    tau({ncol, nlay, this->get_ngpt()}),
    ssa({ncol, nlay, this->get_ngpt()}),
    g  ({ncol, nlay, this->get_ngpt()})
{}


void Optical_props_2str::set_subset(
        const std::unique_ptr<Optical_props_arry>& optical_props_sub,
        const int col_s, const int col_e)
{
    for (int igpt=1; igpt<=tau.dim(3); ++igpt)
        for (int ilay=1; ilay<=tau.dim(2); ++ilay)
            for (int icol=col_s; icol<=col_e; ++icol)
            {
                tau({icol, ilay, igpt}) = optical_props_sub->get_tau()({icol-col_s+1, ilay, igpt});
                ssa({icol, ilay, igpt}) = optical_props_sub->get_ssa()({icol-col_s+1, ilay, igpt});
                g  ({icol, ilay, igpt}) = optical_props_sub->get_g  ()({icol-col_s+1, ilay, igpt});
            }
}


void Optical_props_2str::get_subset(
        const std::unique_ptr<Optical_props_arry>& optical_props_sub,
        const int col_s, const int col_e)
{
    for (int igpt=1; igpt<=tau.dim(3); ++igpt)
        for (int ilay=1; ilay<=tau.dim(2); ++ilay)
            for (int icol=col_s; icol<=col_e; ++icol)
            {
                tau({icol-col_s+1, ilay, igpt}) = optical_props_sub->get_tau()({icol, ilay, igpt});
                ssa({icol-col_s+1, ilay, igpt}) = optical_props_sub->get_ssa()({icol, ilay, igpt});
                g  ({icol-col_s+1, ilay, igpt}) = optical_props_sub->get_g  ()({icol, ilay, igpt});
            }
}


namespace rrtmgp_kernel_launcher
{
    template<typename Float> void increment_1scalar_by_1scalar(
            int ncol, int nlay, int ngpt,
            Array<Float,3>& tau_inout, const Array<Float,3>& tau_in)

    {
        rrtmgp_kernels::rte_increment_1scalar_by_1scalar(
                &ncol, &nlay, &ngpt,
                tau_inout.ptr(), const_cast<Float*>(tau_in.ptr()));
    }

    template<typename Float> void increment_2stream_by_2stream(
            int ncol, int nlay, int ngpt,
            Array<Float,3>& tau_inout, Array<Float,3>& ssa_inout, Array<Float,3>& g_inout,
            const Array<Float,3>& tau_in, const Array<Float,3>& ssa_in, const Array<Float,3>& g_in)
    {
        rrtmgp_kernels::rte_increment_2stream_by_2stream(
                &ncol, &nlay, &ngpt,
                tau_inout.ptr(), ssa_inout.ptr(), g_inout.ptr(),
                const_cast<Float*>(tau_in.ptr()), const_cast<Float*>(ssa_in.ptr()), const_cast<Float*>(g_in.ptr()));
    }

    template<typename Float> void inc_1scalar_by_1scalar_bybnd(
            int ncol, int nlay, int ngpt,
            Array<Float,3>& tau_inout, const Array<Float,3>& tau_in,
            int nbnd, const Array<int,2>& band_lims_gpoint)

    {
        rrtmgp_kernels::rte_inc_1scalar_by_1scalar_bybnd(
                &ncol, &nlay, &ngpt,
                tau_inout.ptr(), const_cast<Float*>(tau_in.ptr()),
                &nbnd, const_cast<int*>(band_lims_gpoint.ptr()));
    }

    template<typename Float> void inc_2stream_by_2stream_bybnd(
            int ncol, int nlay, int ngpt,
            Array<Float,3>& tau_inout, Array<Float,3>& ssa_inout, Array<Float,3>& g_inout,
            const Array<Float,3>& tau_in, const Array<Float,3>& ssa_in, const Array<Float,3>& g_in,
            int nbnd, const Array<int,2>& band_lims_gpoint)

    {
        rrtmgp_kernels::rte_inc_2stream_by_2stream_bybnd(
                &ncol, &nlay, &ngpt,
                tau_inout.ptr(), ssa_inout.ptr(), g_inout.ptr(),
                const_cast<Float*>(tau_in.ptr()), const_cast<Float*>(ssa_in.ptr()), const_cast<Float*>(g_in.ptr()),
                &nbnd, const_cast<int*>(band_lims_gpoint.ptr()));
    }

    template<typename Float> void delta_scale_2str_k(
            int ncol, int nlay, int ngpt,
            Array<Float,3>& tau_inout, Array<Float,3>& ssa_inout, Array<Float,3>& g_inout)
    {
        rrtmgp_kernels::rte_delta_scale_2str_k(
                &ncol, &nlay, &ngpt,
                tau_inout.ptr(), ssa_inout.ptr(), g_inout.ptr());
    }
}


void Optical_props_2str::delta_scale(const Array<Float,3>& forward_frac)
{
    const int ncol = this->get_ncol();
    const int nlay = this->get_nlay();
    const int ngpt = this->get_ngpt();

    rrtmgp_kernel_launcher::delta_scale_2str_k(
            ncol, nlay, ngpt,
            this->get_tau(), this->get_ssa(), this->get_g());
}


void add_to(Optical_props_1scl& op_inout, const Optical_props_1scl& op_in)
{
    const int ncol = op_inout.get_ncol();
    const int nlay = op_inout.get_nlay();
    const int ngpt = op_inout.get_ngpt();

    if (ngpt == op_in.get_ngpt())
    {
        rrtmgp_kernel_launcher::increment_1scalar_by_1scalar(
                ncol, nlay, ngpt,
                op_inout.get_tau(), op_in.get_tau());
    }
    else
    {
        if (op_in.get_ngpt() != op_inout.get_nband())
            throw std::runtime_error("Cannot add optical properties with incompatible band - gpoint combination");

        rrtmgp_kernel_launcher::inc_1scalar_by_1scalar_bybnd(
                ncol, nlay, ngpt,
                op_inout.get_tau(), op_in.get_tau(),
                op_inout.get_nband(), op_inout.get_band_lims_gpoint());
    }
}


void add_to(Optical_props_2str& op_inout, const Optical_props_2str& op_in)
{
    const int ncol = op_inout.get_ncol();
    const int nlay = op_inout.get_nlay();
    const int ngpt = op_inout.get_ngpt();

    if (ngpt == op_in.get_ngpt())
    {
        rrtmgp_kernel_launcher::increment_2stream_by_2stream(
                ncol, nlay, ngpt,
                op_inout.get_tau(), op_inout.get_ssa(), op_inout.get_g(),
                op_in   .get_tau(), op_in   .get_ssa(), op_in   .get_g());
    }
    else
    {
        if (op_in.get_ngpt() != op_inout.get_nband())
            throw std::runtime_error("Cannot add optical properties with incompatible band - gpoint combination");

        rrtmgp_kernel_launcher::inc_2stream_by_2stream_bybnd(
                ncol, nlay, ngpt,
                op_inout.get_tau(), op_inout.get_ssa(), op_inout.get_g(),
                op_in   .get_tau(), op_in   .get_ssa(), op_in   .get_g(),
                op_inout.get_nband(), op_inout.get_band_lims_gpoint());
    }
}
