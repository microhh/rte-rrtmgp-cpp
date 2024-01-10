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

#ifndef GAS_OPTICS_RT_H
#define GAS_OPTICS_RT_H

#include <string>

#include "Array.h"
#include "Optical_props_rt.h"

// Forward declarations.
class Gas_concs_gpu;
class Source_func_lw_rt;

#ifdef USECUDA
class Gas_optics_rt : public Optical_props_rt
{
    public:
        Gas_optics_rt(
                const Array<Float,2>& band_lims_wvn,
                const Array<int,2>& band_lims_gpt) :
            Optical_props_rt(band_lims_wvn, band_lims_gpt)
        {}

        virtual ~Gas_optics_rt() {};

        virtual bool source_is_internal() const = 0;
        virtual bool source_is_external() const = 0;

        virtual Float get_press_ref_min() const = 0;
        virtual Float get_press_ref_max() const = 0;

        virtual Float get_temp_min() const = 0;
        virtual Float get_temp_max() const = 0;

        // Longwave variant.
        virtual void gas_optics(
                const int igpt, const int col_s, const int ncol_block, const int ncol,
                const Array_gpu<Float,2>& play,
                const Array_gpu<Float,2>& plev,
                const Array_gpu<Float,2>& tlay,
                const Array_gpu<Float,1>& tsfc,
                const Gas_concs_gpu& gas_desc,
                std::unique_ptr<Optical_props_arry_rt>& optical_props,
                Source_func_lw_rt& sources,
                const Array_gpu<Float,2>& col_dry,
                const Array_gpu<Float,2>& tlev) = 0;

        // Shortwave variant.
        virtual void gas_optics(
                const int igpt, const int col_s, const int ncol_block, const int ncol,
                const Array_gpu<Float,2>& play,
                const Array_gpu<Float,2>& plev,
                const Array_gpu<Float,2>& tlay,
                const Gas_concs_gpu& gas_desc,
                std::unique_ptr<Optical_props_arry_rt>& optical_props,
                Array_gpu<Float,1>& toa_src,
                const Array_gpu<Float,2>& col_dry) = 0;

       virtual Float get_tsi() const = 0;
       
       virtual Float band_source(const int gpt_start, const int gpt_end) const = 0;
};
#endif

#endif
