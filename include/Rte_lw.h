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

#ifndef RTE_LW_H
#define RTE_LW_H

#include <memory>

#include "types.h"


// Forward declarations.
template<typename, int> class Array;
template<typename, int> class Array_gpu;
class Optical_props_arry;
class Optical_props_arry_gpu;
class Source_func_lw;
class Source_func_lw_gpu;


class Rte_lw
{
    public:
        static void rte_lw(
                const std::unique_ptr<Optical_props_arry>& optical_props,
                const Bool top_at_1,
                const Source_func_lw& sources,
                const Array<Float,2>& sfc_emis,
                const Array<Float,2>& inc_flux,
                Array<Float,3>& gpt_flux_up,
                Array<Float,3>& gpt_flux_dn,
                const int n_gauss_angles);

        static void expand_and_transpose(
                const std::unique_ptr<Optical_props_arry>& ops,
                const Array<Float,2> arr_in,
                Array<Float,2>& arr_out);
};

#ifdef USECUDA
class Rte_lw_gpu
{
    public:
        void rte_lw(
                const std::unique_ptr<Optical_props_arry_gpu>& optical_props,
                const bool top_at_1,
                const Source_func_lw_gpu& sources,
                const Array_gpu<Float,2>& sfc_emis,
                const Array_gpu<Float,2>& inc_flux,
                Array_gpu<Float,3>& gpt_flux_up,
                Array_gpu<Float,3>& gpt_flux_dn,
                const int n_gauss_angles);

        void expand_and_transpose(
                const std::unique_ptr<Optical_props_arry_gpu>& ops,
                const Array_gpu<Float,2> arr_in,
                Array_gpu<Float,2>& arr_out);

};
#endif

#endif
