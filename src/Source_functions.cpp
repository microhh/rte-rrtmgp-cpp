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

#include "Source_functions.h"
#include "Array.h"
#include "Optical_props.h"


Source_func_lw::Source_func_lw(
        const int n_col,
        const int n_lay,
        const Optical_props& optical_props) :
    Optical_props(optical_props),
    sfc_source({n_col, optical_props.get_ngpt()}),
    sfc_source_jac({n_col, optical_props.get_ngpt()}),
    lay_source({n_col, n_lay, optical_props.get_ngpt()}),
    lev_source({n_col, n_lay+1, optical_props.get_ngpt()})
{}


void Source_func_lw::set_subset(
        const Source_func_lw& sources_sub,
        const int col_s, const int col_e)
{
    for (int igpt=1; igpt<=lay_source.dim(3); ++igpt)
        for (int icol=col_s; icol<=col_e; ++icol)
            sfc_source({icol, igpt}) = sources_sub.get_sfc_source()({icol-col_s+1, igpt});

    for (int igpt=1; igpt<=lay_source.dim(3); ++igpt)
        for (int ilay=1; ilay<=lay_source.dim(2); ++ilay)
            for (int icol=col_s; icol<=col_e; ++icol)
            {
                lay_source({icol, ilay, igpt}) = sources_sub.get_lay_source()({icol-col_s+1, ilay, igpt});
                lev_source({icol, ilay, igpt}) = sources_sub.get_lev_source()({icol-col_s+1, ilay, igpt});
            }

    const int nlev = lev_source.dim(2);

    for (int igpt=1; igpt<=lay_source.dim(3); ++igpt)
        for (int icol=col_s; icol<=col_e; ++icol)
            lev_source({icol, nlev, igpt}) = sources_sub.get_lev_source()({icol-col_s+1, nlev, igpt});
}


void Source_func_lw::get_subset(
        const Source_func_lw& sources_sub,
        const int col_s, const int col_e)
{
    for (int igpt=1; igpt<=lay_source.dim(3); ++igpt)
        for (int icol=col_s; icol<=col_e; ++icol)
            sfc_source({icol-col_s+1, igpt}) = sources_sub.get_sfc_source()({icol, igpt});

    for (int igpt=1; igpt<=lay_source.dim(3); ++igpt)
        for (int ilay=1; ilay<=lay_source.dim(2); ++ilay)
            for (int icol=col_s; icol<=col_e; ++icol)
            {
                lay_source({icol-col_s+1, ilay, igpt}) = sources_sub.get_lay_source()({icol, ilay, igpt});
                lev_source({icol-col_s+1, ilay, igpt}) = sources_sub.get_lev_source()({icol, ilay, igpt});
            }

    const int nlev = lev_source.dim(2);

    for (int igpt=1; igpt<=lay_source.dim(3); ++igpt)
        for (int icol=col_s; icol<=col_e; ++icol)
            lev_source({icol-col_s+1, nlev, igpt}) = sources_sub.get_lev_source()({icol, nlev, igpt});
}
