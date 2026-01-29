/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/RobertPincus/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/microhh/rte-rrtmgp-cpp
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

#ifndef RTE_KERNELS_CUDA_RT_H
#define RTE_KERNELS_CUDA_RT_H


#include "types.h"


namespace Rte_solver_kernels_cuda_rt
{
    void apply_BC(
            const int ncol, const int nlay, const Bool top_at_1,
            const Float* inc_flux_dir, const Float* mu0, Float* gpt_flux_dir);

    void apply_BC(const int ncol, const int nlay, const Bool top_at_1, Float* gpt_flux_dn);

    void apply_BC(const int ncol, const int nlay, const Bool top_at_1, const Float* inc_flux_dif, Float* gpt_flux_dn);

    void apply_BC(const int ncol, const int nlay, const Bool top_at_1, const Float inc_flux, Float* gpt_flux_dn);

    void sw_solver_2stream(
            const int ncol, const int nlay, const Bool top_at_1,
            const Float* tau, const Float* ssa, const Float* g,
            const Float* mu0, const Float* sfc_alb_dir, const Float* sfc_alb_dif,
            Float* flux_up, Float* flux_dn, Float* flux_dir);

    void lw_solver_noscat_gaussquad(
            const int ncol, const int nlay, const Bool top_at_1, const int nmus,
            const Float* ds, const Float* weights, const Float* tau, const Float* lay_source,
            const Float* lev_source, const Float* sfc_emis,
            const Float* sfc_src, Float* flux_up, Float* flux_dn,
            const Float* sfc_src_jac, Float* flux_up_jac);

    void lw_solver_2stream(const int ncol, const int nlay, const Bool top_at_1,
             const Float* tau, const Float* ssa, const Float* g,
             const Float* lay_source, const Float* lev_source,
             const Float* sfc_emis, const Float* sfc_src,
             Float* flux_up, Float* flux_dn);
}
#endif
