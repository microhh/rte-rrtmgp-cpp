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

#ifndef GPT_COMBINE_KERNELS_CUDA_RT_H
#define GPT_COMBINE_KERNELS_CUDA_RT_H

#include "types.h"
// #include "gas_concs.h"


namespace Gpt_combine_kernels_cuda_rt
{

    void get_from_gpoint(const int ncol, const int igpt,
                  Float* var_full, const Float* var_sub);

    void add_from_gpoint(const int ncol, const int nlay,
                  Float* var1_full, Float* var2_full, Float* var3_full,  Float* var4_full, Float* var5_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub, const Float* var4_sub, const Float* var5_sub);

    void add_from_gpoint(const int ncol, const int nlay,
                  Float* var1_full, Float* var2_full, Float* var3_full,  Float* var4_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub, const Float* var4_sub);

    void add_from_gpoint(const int ncol, const int nlay,
                  Float* var1_full, Float* var2_full,
                  const Float* var1_sub, const Float* var2_sub);

    void add_from_gpoint(const int ncol, const int nlay,
                  Float* var1_full, const Float* var1_sub);

    void add_from_gpoint(const int ncol, const int nlay,
                  Float* var1_full, Float* var2_full, Float* var3_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub);


    void get_from_gpoint(const int ncol, const int nlay, const int igpt,
                  Float* var1_full, Float* var2_full, Float* var3_full,  Float* var4_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub, const Float* var4_sub);


    void get_from_gpoint(const int ncol, const int nlay, const int igpt,
                  Float* var1_full, Float* var2_full, Float* var3_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub);
}
#endif
