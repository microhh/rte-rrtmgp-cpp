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

#include "Gas_concs_rt.h"
#include "Array.h"

Gas_concs_rt::Gas_concs_rt(const Gas_concs& gas_concs_ref)
{
    // The emplace function should call the appropriate constructor of Array_gpu.
    for (auto& g : gas_concs_ref.gas_concs_map)
        this->gas_concs_map.emplace(g.first, g.second);
}

Gas_concs_rt::Gas_concs_rt(const Gas_concs_rt& gas_concs_ref, const int start, const int size)
{
    const int end = start + size - 1;
    for (auto& g : gas_concs_ref.gas_concs_map)
    {
        if (g.second.dim(1) == 1)
            this->gas_concs_map.emplace(g.first, g.second);
        else
        {
            Array_gpu<Float,2> gas_conc_subset = g.second.subset({{ {start, end}, {1, g.second.dim(2)} }});
            this->gas_concs_map.emplace(g.first, gas_conc_subset);
        }
    }
}

Gas_concs_rt::~Gas_concs_rt()
{
}

// Get gas from map.
const Array_gpu<Float,2>& Gas_concs_rt::get_vmr(const std::string& name) const
{
    return this->gas_concs_map.at(name);
}

void Gas_concs_gpu::set_vmr(const std::string& name, const Array_gpu<Float,2>& data)
{
    gas_concs_map.erase(name);
    gas_concs_map.emplace(name, data);
}

void Gas_concs_gpu::set_vmr(const std::string& name, const Array<Float,2>& data)
{
    Array_gpu<Float,2> tmp(data);
    gas_concs_map.at(name) = tmp;
}

// Check if gas exists in map.
bool Gas_concs_rt::exists(const std::string& name) const
{
    return gas_concs_map.count(name) != 0;
}

