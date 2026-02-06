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
#include "fluxes_kernels_cuda.h"


Fluxes_broadband_gpu::Fluxes_broadband_gpu(const int ncol, const int nlev) :
    flux_up    ({ncol, nlev}),
    flux_dn    ({ncol, nlev}),
    flux_dn_dir({ncol, nlev}),
    flux_net   ({ncol, nlev})
{}


void Fluxes_broadband_gpu::reduce(
    const Array_gpu<Float,3>& gpt_flux_up, const Array_gpu<Float,3>& gpt_flux_dn,
    const std::unique_ptr<Optical_props_arry_gpu>& spectral_disc,
    const bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = gpt_flux_up.dim(3);

    Fluxes_kernels_cuda::sum_broadband(ncol, nlev, ngpt, gpt_flux_up.ptr(), this->flux_up.ptr());
    Fluxes_kernels_cuda::sum_broadband(ncol, nlev, ngpt, gpt_flux_dn.ptr(), this->flux_dn.ptr());

    Fluxes_kernels_cuda::net_broadband_precalc(
            ncol, nlev, this->flux_dn.ptr(), this->flux_up.ptr(), this->flux_net.ptr());
}


//// CvH: unnecessary code duplication.
void Fluxes_broadband_gpu::reduce(
    const Array_gpu<Float,3>& gpt_flux_up, const Array_gpu<Float,3>& gpt_flux_dn, const Array_gpu<Float,3>& gpt_flux_dn_dir,
    const std::unique_ptr<Optical_props_arry_gpu>& spectral_disc,
    const bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = gpt_flux_up.dim(3);

    reduce(gpt_flux_up, gpt_flux_dn, spectral_disc, top_at_1);

    Fluxes_kernels_cuda::sum_broadband(ncol, nlev, ngpt, gpt_flux_dn_dir.ptr(), this->flux_dn_dir.ptr());
}


Fluxes_byband_gpu::Fluxes_byband_gpu(const int ncol, const int nlev, const int nbnd) :
    Fluxes_broadband_gpu(ncol, nlev),
    bnd_flux_up    ({ncol, nlev, nbnd}),
    bnd_flux_dn    ({ncol, nlev, nbnd}),
    bnd_flux_dn_dir({ncol, nlev, nbnd}),
    bnd_flux_net   ({ncol, nlev, nbnd})
{}


void Fluxes_byband_gpu::reduce(
    const Array_gpu<Float,3>& gpt_flux_up,
    const Array_gpu<Float,3>& gpt_flux_dn,
    const std::unique_ptr<Optical_props_arry_gpu>& spectral_disc,
    const bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = spectral_disc->get_ngpt();
    const int nbnd = spectral_disc->get_nband();

    const Array_gpu<int,2>& band_lims = spectral_disc->get_band_lims_gpoint();

    Fluxes_broadband_gpu::reduce(
            gpt_flux_up, gpt_flux_dn,
            spectral_disc, top_at_1);

    Fluxes_kernels_cuda::sum_byband(
            ncol, nlev, ngpt, nbnd, band_lims.ptr(),
            gpt_flux_up.ptr(), this->bnd_flux_up.ptr());

    Fluxes_kernels_cuda::sum_byband(
            ncol, nlev, ngpt, nbnd, band_lims.ptr(),
            gpt_flux_dn.ptr(), this->bnd_flux_dn.ptr());

    Fluxes_kernels_cuda::net_byband_full(
            ncol, nlev, ngpt, nbnd, band_lims.ptr(),
            this->bnd_flux_dn.ptr(), this->bnd_flux_up.ptr(), this->bnd_flux_net.ptr());
}


// CvH: a lot of code duplication.
void Fluxes_byband_gpu::reduce(
    const Array_gpu<Float,3>& gpt_flux_up,
    const Array_gpu<Float,3>& gpt_flux_dn,
    const Array_gpu<Float,3>& gpt_flux_dn_dir,
    const std::unique_ptr<Optical_props_arry_gpu>& spectral_disc,
    const bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = spectral_disc->get_ngpt();
    const int nbnd = spectral_disc->get_nband();

    const Array_gpu<int,2>& band_lims = spectral_disc->get_band_lims_gpoint();

    Fluxes_broadband_gpu::reduce(
            gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir,
            spectral_disc, top_at_1);

    reduce(gpt_flux_up, gpt_flux_dn, spectral_disc, top_at_1);

    Fluxes_kernels_cuda::sum_byband(
            ncol, nlev, ngpt, nbnd, band_lims.ptr(),
            gpt_flux_dn_dir.ptr(), this->bnd_flux_dn_dir.ptr());
}
