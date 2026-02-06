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

#ifndef SOURCE_FUNCTIONS_RT_H
#define SOURCE_FUNCTIONS_RT_H
#include "optical_props_rt.h"

template<typename, int> class Array_gpu;

#ifdef USECUDA
class Source_func_lw_rt : public Optical_props_rt
{
    public:
        Source_func_lw_rt(
                const int n_col,
                const int n_lay,
                const Optical_props_rt& optical_props);

//        void set_subset(
//                const Source_func_lw& sources_sub,
//                const int col_s, const int col_e);
//
//        void get_subset(
//                const Source_func_lw& sources_sub,
//                const int col_s, const int col_e);
//
        Array_gpu<Float,1>& get_sfc_source()     { return sfc_source;     }
        Array_gpu<Float,1>& get_sfc_source_jac() { return sfc_source_jac; }
        Array_gpu<Float,2>& get_lay_source()     { return lay_source;     }
        Array_gpu<Float,2>& get_lev_source()     { return lev_source; }

        const Array_gpu<Float,1>& get_sfc_source()     const { return sfc_source;     }
        const Array_gpu<Float,1>& get_sfc_source_jac() const { return sfc_source_jac; }
        const Array_gpu<Float,2>& get_lay_source()     const { return lay_source;     }
        const Array_gpu<Float,2>& get_lev_source()     const { return lev_source; }

    private:
        Array_gpu<Float,1> sfc_source;
        Array_gpu<Float,1> sfc_source_jac;
        Array_gpu<Float,2> lay_source;
        Array_gpu<Float,2> lev_source;
};

#endif
#endif
