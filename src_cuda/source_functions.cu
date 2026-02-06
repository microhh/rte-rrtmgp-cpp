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

#include "source_functions.h"
#include "array.h"
#include "optical_props.h"


Source_func_lw_gpu::Source_func_lw_gpu(
        const int n_col,
        const int n_lay,
        const Optical_props_gpu& optical_props) :
    Optical_props_gpu(optical_props),
    sfc_source({n_col, optical_props.get_ngpt()}),
    sfc_source_jac({n_col, optical_props.get_ngpt()}),
    lay_source({n_col, n_lay, optical_props.get_ngpt()}),
    lev_source({n_col, n_lay+1, optical_props.get_ngpt()})
{}
