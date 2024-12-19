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

#ifndef CLOUD_OPTICS_RT_H
#define CLOUD_OPTICS_RT_H

#include "Array.h"
#include "Optical_props_rt.h"
#include "types.h"

// Forward declarations.
class Optical_props_rt;

#ifdef USECUDA
class Cloud_optics_rt : public Optical_props_rt
{
    public:
        Cloud_optics_rt(
                const Array<Float,2>& band_lims_wvn,
                const Float radliq_lwr, const Float radliq_upr, const Float radliq_fac,
                const Float diamice_lwr, const Float diamice_upr, const Float diamice_fac,
                const Array<Float,2>& lut_extliq, const Array<Float,2>& lut_ssaliq, const Array<Float,2>& lut_asyliq,
                const Array<Float,3>& lut_extice, const Array<Float,3>& lut_ssaice, const Array<Float,3>& lut_asyice);

        void cloud_optics(
                const int ibnd,
                const Array_gpu<Float,2>& clwp, const Array_gpu<Float,2>& ciwp,
                const Array_gpu<Float,2>& reliq, const Array_gpu<Float,2>& deice,
                Optical_props_1scl_rt& optical_props);

        void cloud_optics(
                const int ibnd,
                const Array_gpu<Float,2>& clwp, const Array_gpu<Float,2>& ciwp,
                const Array_gpu<Float,2>& reliq, const Array_gpu<Float,2>& deice,
                Optical_props_2str_rt& optical_props);

    private:
        int liq_nsteps;
        int ice_nsteps;
        Float liq_step_size;
        Float ice_step_size;

        // Lookup table constants.
        Float radliq_lwr;
        Float radliq_upr;
        Float diamice_lwr;
        Float diamice_upr;

        // Lookup table coefficients.
        Array<Float,2> lut_extliq;
        Array<Float,2> lut_ssaliq;
        Array<Float,2> lut_asyliq;
        Array<Float,2> lut_extice;
        Array<Float,2> lut_ssaice;
        Array<Float,2> lut_asyice;

        // gpu versions
        Array_gpu<Float,2> lut_extliq_gpu;
        Array_gpu<Float,2> lut_ssaliq_gpu;
        Array_gpu<Float,2> lut_asyliq_gpu;
        Array_gpu<Float,2> lut_extice_gpu;
        Array_gpu<Float,2> lut_ssaice_gpu;
        Array_gpu<Float,2> lut_asyice_gpu;
};
#endif

#endif
