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

#include "fluxes.h"
#include "array.h"
#include "optical_props.h"

#include "rrtmgp_kernels.h"


namespace rrtmgp_kernel_launcher
{
    template<typename Float>
    void sum_broadband(
            int ncol, int nlev, int ngpt,
            const Array<Float,3>& spectral_flux, Array<Float,2>& broadband_flux)
    {
        rrtmgp_kernels::rte_sum_broadband(
                &ncol, &nlev, &ngpt,
                const_cast<Float*>(spectral_flux.ptr()),
                broadband_flux.ptr());
    }

    template<typename Float>
    void net_broadband(
            int ncol, int nlev,
            const Array<Float,2>& broadband_flux_dn, const Array<Float,2>& broadband_flux_up,
            Array<Float,2>& broadband_flux_net)
    {
        rrtmgp_kernels::rte_net_broadband_precalc(
                &ncol, &nlev,
                const_cast<Float*>(broadband_flux_dn.ptr()),
                const_cast<Float*>(broadband_flux_up.ptr()),
                broadband_flux_net.ptr());
    }

    template<typename Float>
    void sum_byband(
            int ncol, int nlev, int ngpt, int nbnd,
            const Array<int,2>& band_lims,
            const Array<Float,3>& spectral_flux,
            Array<Float,3>& byband_flux)
    {
        rrtmgp_kernels::sum_byband(
                &ncol, &nlev, &ngpt, &nbnd,
                const_cast<int*>(band_lims.ptr()),
                const_cast<Float*>(spectral_flux.ptr()),
                byband_flux.ptr());
    }

    template<typename Float>
    void net_byband(
            int ncol, int nlev, int nband,
            const Array<Float,3>& byband_flux_dn, const Array<Float,3>& byband_flux_up,
            Array<Float,3>& byband_flux_net)
    {
        rrtmgp_kernels::net_byband_precalc(
                &ncol, &nlev, &nband,
                const_cast<Float*>(byband_flux_dn.ptr()),
                const_cast<Float*>(byband_flux_up.ptr()),
                byband_flux_net.ptr());
    }
}


Fluxes_broadband::Fluxes_broadband(const int ncol, const int nlev) :
    flux_up    ({ncol, nlev}),
    flux_dn    ({ncol, nlev}),
    flux_dn_dir({ncol, nlev}),
    flux_net   ({ncol, nlev})
{}


void Fluxes_broadband::reduce(
    const Array<Float,3>& gpt_flux_up, const Array<Float,3>& gpt_flux_dn,
    const std::unique_ptr<Optical_props_arry>& spectral_disc,
    const Bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = gpt_flux_up.dim(3);

    rrtmgp_kernel_launcher::sum_broadband(
            ncol, nlev, ngpt, gpt_flux_up, this->flux_up);

    rrtmgp_kernel_launcher::sum_broadband(
            ncol, nlev, ngpt, gpt_flux_dn, this->flux_dn);

    rrtmgp_kernel_launcher::net_broadband(
            ncol, nlev, this->flux_dn, this->flux_up, this->flux_net);
}


// CvH: unnecessary code duplication.
void Fluxes_broadband::reduce(
    const Array<Float,3>& gpt_flux_up, const Array<Float,3>& gpt_flux_dn, const Array<Float,3>& gpt_flux_dn_dir,
    const std::unique_ptr<Optical_props_arry>& spectral_disc,
    const Bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = gpt_flux_up.dim(3);

    reduce(gpt_flux_up, gpt_flux_dn, spectral_disc, top_at_1);

    rrtmgp_kernel_launcher::sum_broadband(
            ncol, nlev, ngpt,
            gpt_flux_dn_dir, this->flux_dn_dir);
}


Fluxes_byband::Fluxes_byband(const int ncol, const int nlev, const int nbnd) :
    Fluxes_broadband(ncol, nlev),
    bnd_flux_up    ({ncol, nlev, nbnd}),
    bnd_flux_dn    ({ncol, nlev, nbnd}),
    bnd_flux_dn_dir({ncol, nlev, nbnd}),
    bnd_flux_net   ({ncol, nlev, nbnd})
{}


void Fluxes_byband::reduce(
    const Array<Float,3>& gpt_flux_up,
    const Array<Float,3>& gpt_flux_dn,
    const std::unique_ptr<Optical_props_arry>& spectral_disc,
    const Bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = spectral_disc->get_ngpt();
    const int nbnd = spectral_disc->get_nband();

    const Array<int,2>& band_lims = spectral_disc->get_band_lims_gpoint();

    Fluxes_broadband::reduce(
            gpt_flux_up, gpt_flux_dn,
            spectral_disc, top_at_1);

    rrtmgp_kernel_launcher::sum_byband(
            ncol, nlev, ngpt, nbnd, band_lims,
            gpt_flux_up, this->bnd_flux_up);

    rrtmgp_kernel_launcher::sum_byband(
            ncol, nlev, ngpt, nbnd, band_lims,
            gpt_flux_dn, this->bnd_flux_dn);

    rrtmgp_kernel_launcher::net_byband(
            ncol, nlev, nbnd,
            this->bnd_flux_dn, this->bnd_flux_up, this->bnd_flux_net);
}


// CvH: a lot of code duplication.
void Fluxes_byband::reduce(
    const Array<Float,3>& gpt_flux_up,
    const Array<Float,3>& gpt_flux_dn,
    const Array<Float,3>& gpt_flux_dn_dir,
    const std::unique_ptr<Optical_props_arry>& spectral_disc,
    const Bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = spectral_disc->get_ngpt();
    const int nbnd = spectral_disc->get_nband();

    const Array<int,2>& band_lims = spectral_disc->get_band_lims_gpoint();

    Fluxes_broadband::reduce(
            gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir,
            spectral_disc, top_at_1);

    reduce(gpt_flux_up, gpt_flux_dn, spectral_disc, top_at_1);

    rrtmgp_kernel_launcher::sum_byband(
            ncol, nlev, ngpt, nbnd, band_lims,
            gpt_flux_dn_dir, this->bnd_flux_dn_dir);
}
