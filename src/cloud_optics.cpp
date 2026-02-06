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

#include <limits>
#include "cloud_optics.h"


Cloud_optics::Cloud_optics(
        const Array<Float,2>& band_lims_wvn,
        const Float radliq_lwr, const Float radliq_upr, const Float radliq_fac,
        const Float diamice_lwr, const Float diamice_upr, const Float diamice_fac,
        const Array<Float,2>& lut_extliq, const Array<Float,2>& lut_ssaliq, const Array<Float,2>& lut_asyliq,
        const Array<Float,3>& lut_extice, const Array<Float,3>& lut_ssaice, const Array<Float,3>& lut_asyice) :
    Optical_props(band_lims_wvn)
{
    const int nsize_liq = lut_extliq.dim(1);
    const int nsize_ice = lut_extice.dim(1);

    this->liq_nsteps = nsize_liq;
    this->ice_nsteps = nsize_ice;
    this->liq_step_size = (radliq_upr - radliq_lwr) / (nsize_liq - Float(1.));
    this->ice_step_size = (diamice_upr - diamice_lwr) / (nsize_ice - Float(1.));

    // Load LUT constants.
    this->radliq_lwr = radliq_lwr;
    this->radliq_upr = radliq_upr;
    this->diamice_lwr = diamice_lwr;
    this->diamice_upr = diamice_upr;

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
}


void compute_all_from_table(
        const int ncol, const int nlay, const int nbnd, const Array<Bool,2>& mask,
        const Array<Float,2>& cwp, const Array<Float,2>& re,
        const int nsteps, const Float step_size, const Float offset,
        const Array<Float,2>& tau_table, const Array<Float,2>& ssa_table, const Array<Float,2>& asy_table,
        Array<Float,3>& tau, Array<Float,3>& taussa, Array<Float,3>& taussag)
{
    for (int ibnd=1; ibnd<=nbnd; ++ibnd)
        for (int ilay=1; ilay<=nlay; ++ilay)
            for (int icol=1; icol<=ncol; ++icol)
            {
                if (mask({icol, ilay}))
                {
                    const int index = std::min(
                            static_cast<int>((re({icol, ilay}) - offset) / step_size)+1, nsteps-1);
                    const Float fint = (re({icol, ilay}) - offset) / step_size - (index-1);

                    const Float tau_local = cwp({icol, ilay}) *
                        (tau_table({index, ibnd}) + fint * (tau_table({index+1, ibnd}) - tau_table({index, ibnd})));
                    const Float taussa_local = tau_local *
                        (ssa_table({index, ibnd}) + fint * (ssa_table({index+1, ibnd}) - ssa_table({index, ibnd})));
                    const Float taussag_local = taussa_local *
                        (asy_table({index, ibnd}) + fint * (asy_table({index+1, ibnd}) - asy_table({index, ibnd})));

                    tau    ({icol, ilay, ibnd}) = tau_local;
                    taussa ({icol, ilay, ibnd}) = taussa_local;
                    taussag({icol, ilay, ibnd}) = taussag_local;
                }
                else
                {
                    tau    ({icol, ilay, ibnd}) = Float(0.);
                    taussa ({icol, ilay, ibnd}) = Float(0.);
                    taussag({icol, ilay, ibnd}) = Float(0.);
                }
            }
}


// Two-stream variant of cloud optics.
void Cloud_optics::cloud_optics(
        const Array<Float,2>& clwp, const Array<Float,2>& ciwp,
        const Array<Float,2>& reliq, const Array<Float,2>& deice,
        Optical_props_2str& optical_props)
{
    const int ncol = clwp.dim(1);
    const int nlay = clwp.dim(2);
    const int nbnd = this->get_nband();

    Optical_props_2str clouds_liq(ncol, nlay, optical_props);
    Optical_props_2str clouds_ice(ncol, nlay, optical_props);

    // Set the mask.
    constexpr Float mask_min_value = Float(0.);
    Array<Bool,2> liqmsk({ncol, nlay});
    for (int i=0; i<liqmsk.size(); ++i)
        liqmsk.v()[i] = clwp.v()[i] > mask_min_value;

    Array<Bool,2> icemsk({ncol, nlay});
    for (int i=0; i<icemsk.size(); ++i)
        icemsk.v()[i] = ciwp.v()[i] > mask_min_value;

    // Temporary arrays for storage.
    Array<Float,3> ltau    ({ncol, nlay, nbnd});
    Array<Float,3> ltaussa ({ncol, nlay, nbnd});
    Array<Float,3> ltaussag({ncol, nlay, nbnd});

    Array<Float,3> itau    ({ncol, nlay, nbnd});
    Array<Float,3> itaussa ({ncol, nlay, nbnd});
    Array<Float,3> itaussag({ncol, nlay, nbnd});

    // Liquid water.
    compute_all_from_table(
            ncol, nlay, nbnd, liqmsk, clwp, reliq,
            this->liq_nsteps, this->liq_step_size, this->radliq_lwr,
            this->lut_extliq, this->lut_ssaliq, this->lut_asyliq,
            ltau, ltaussa, ltaussag);

    // Ice.
    compute_all_from_table(
            ncol, nlay, nbnd, icemsk, ciwp, deice,
            this->ice_nsteps, this->ice_step_size, this->diamice_lwr,
            this->lut_extice, this->lut_ssaice, this->lut_asyice,
            itau, itaussa, itaussag);

    constexpr Float eps = std::numeric_limits<Float>::epsilon();

    // Process the calculated optical properties.
    for (int ibnd=1; ibnd<=nbnd; ++ibnd)
        for (int ilay=1; ilay<=nlay; ++ilay)
            #pragma ivdep
            for (int icol=1; icol<=ncol; ++icol)
            {
                const Float tau = ltau({icol, ilay, ibnd}) + itau({icol, ilay, ibnd});
                const Float taussa = ltaussa({icol, ilay, ibnd}) + itaussa({icol, ilay, ibnd});
                const Float taussag = ltaussag({icol, ilay, ibnd}) + itaussag({icol, ilay, ibnd});

                optical_props.get_tau()({icol, ilay, ibnd}) = tau;
                optical_props.get_ssa()({icol, ilay, ibnd}) = taussa / std::max(tau, eps);
                optical_props.get_g  ()({icol, ilay, ibnd}) = taussag / std::max(taussa, eps);
            }
}


// 1scl variant of cloud optics.
void Cloud_optics::cloud_optics(
        const Array<Float,2>& clwp, const Array<Float,2>& ciwp,
        const Array<Float,2>& reliq, const Array<Float,2>& deice,
        Optical_props_1scl& optical_props)
{
    const int ncol = clwp.dim(1);
    const int nlay = clwp.dim(2);
    const int nbnd = this->get_nband();

    Optical_props_1scl clouds_liq(ncol, nlay, optical_props);
    Optical_props_1scl clouds_ice(ncol, nlay, optical_props);

    // Set the mask.
    constexpr Float mask_min_value = static_cast<Float>(0.);
    Array<Bool,2> liqmsk({ncol, nlay});
    for (int i=0; i<liqmsk.size(); ++i)
        liqmsk.v()[i] = clwp.v()[i] > mask_min_value;

    Array<Bool,2> icemsk({ncol, nlay});
    for (int i=0; i<icemsk.size(); ++i)
        icemsk.v()[i] = ciwp.v()[i] > mask_min_value;

    // Temporary arrays for storage.
    Array<Float,3> ltau    ({ncol, nlay, nbnd});
    Array<Float,3> ltaussa ({ncol, nlay, nbnd});
    Array<Float,3> ltaussag({ncol, nlay, nbnd});

    Array<Float,3> itau    ({ncol, nlay, nbnd});
    Array<Float,3> itaussa ({ncol, nlay, nbnd});
    Array<Float,3> itaussag({ncol, nlay, nbnd});

    // Liquid water.
    compute_all_from_table(
            ncol, nlay, nbnd, liqmsk, clwp, reliq,
            this->liq_nsteps, this->liq_step_size, this->radliq_lwr,
            this->lut_extliq, this->lut_ssaliq, this->lut_asyliq,
            ltau, ltaussa, ltaussag);

    // Ice.
    compute_all_from_table(
            ncol, nlay, nbnd, icemsk, ciwp, deice,
            this->ice_nsteps, this->ice_step_size, this->diamice_lwr,
            this->lut_extice, this->lut_ssaice, this->lut_asyice,
            itau, itaussa, itaussag);

    // Process the calculated optical properties.
    for (int ibnd=1; ibnd<=nbnd; ++ibnd)
        for (int ilay=1; ilay<=nlay; ++ilay)
            #pragma ivdep
            for (int icol=1; icol<=ncol; ++icol)
            {
                const Float tau = (ltau({icol, ilay, ibnd}) - ltaussa({icol, ilay, ibnd}))
                             + (itau({icol, ilay, ibnd}) - itaussa({icol, ilay, ibnd}));

                optical_props.get_tau()({icol, ilay, ibnd}) = tau;
            }
}
