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

#ifndef FLUXES_RT_H
#define FLUXES_RT_H

#include <memory>
#include <stdexcept>

#include "types.h"
#include "array.h"

// Forward declarations.
class Optical_props_arry_rt;


//#ifdef USECUDA

class Fluxes_rt
{
    public:
        virtual void reduce(
                const Array_gpu<Float,3>& gpt_flux_up,
                const Array_gpu<Float,3>& gpt_flux_dn,
                const std::unique_ptr<Optical_props_arry_rt>& optical_props,
                const bool top_at_1) = 0;

        virtual void reduce(
                const Array_gpu<Float,3>& gpt_flux_up,
                const Array_gpu<Float,3>& gpt_flux_dn,
                const Array_gpu<Float,3>& gpt_flux_dn_dir,
                const std::unique_ptr<Optical_props_arry_rt>& optical_props,
                const bool top_at_1) = 0;
};


class Fluxes_broadband_rt : public Fluxes_rt
{
    public:
        Fluxes_broadband_rt(const int ncol_x, const int ncol_y, const int n_z, const int nlev);
        virtual ~Fluxes_broadband_rt() {};

        virtual void net_flux();

        virtual void reduce(
                const Array_gpu<Float,3>& gpt_flux_up,
                const Array_gpu<Float,3>& gpt_flux_dn,
                const std::unique_ptr<Optical_props_arry_rt>& optical_props,
                const bool top_at_1);

        virtual void reduce(
                const Array_gpu<Float,3>& gpt_flux_up,
                const Array_gpu<Float,3>& gpt_flux_dn,
                const Array_gpu<Float,3>& gpt_flux_dn_dir,
                const std::unique_ptr<Optical_props_arry_rt>& optical_props,
                const bool top_at_1);

        Array_gpu<Float,2>& get_flux_up    () { return flux_up;     }
        Array_gpu<Float,2>& get_flux_dn    () { return flux_dn;     }
        Array_gpu<Float,2>& get_flux_dn_dir() { return flux_dn_dir; }
        Array_gpu<Float,2>& get_flux_net   () { return flux_net;    }

        Array_gpu<Float,2>& get_flux_tod_dn    () { return flux_tod_dn; }
        Array_gpu<Float,2>& get_flux_tod_up    () { return flux_tod_up; }
        Array_gpu<Float,2>& get_flux_sfc_dir   () { return flux_sfc_dir;}
        Array_gpu<Float,2>& get_flux_sfc_dif   () { return flux_sfc_dif;}
        Array_gpu<Float,2>& get_flux_sfc_up    () { return flux_sfc_up; }
        Array_gpu<Float,3>& get_flux_abs_dir   () { return flux_abs_dir;}
        Array_gpu<Float,3>& get_flux_abs_dif   () { return flux_abs_dif;}


        virtual Array_gpu<Float,3>& get_bnd_flux_up    () { throw std::runtime_error("Band fluxes are not available"); }
        virtual Array_gpu<Float,3>& get_bnd_flux_dn    () { throw std::runtime_error("Band fluxes are not available"); }
        virtual Array_gpu<Float,3>& get_bnd_flux_dn_dir() { throw std::runtime_error("Band fluxes are not available"); }
        virtual Array_gpu<Float,3>& get_bnd_flux_net   () { throw std::runtime_error("Band fluxes are not available"); }

    private:
        Array_gpu<Float,2> flux_up;
        Array_gpu<Float,2> flux_dn;
        Array_gpu<Float,2> flux_dn_dir;
        Array_gpu<Float,2> flux_net;
        Array_gpu<Float,2> flux_tod_dn;
        Array_gpu<Float,2> flux_tod_up;
        Array_gpu<Float,2> flux_sfc_dir;
        Array_gpu<Float,2> flux_sfc_dif;
        Array_gpu<Float,2> flux_sfc_up;
        Array_gpu<Float,3> flux_abs_dir;
        Array_gpu<Float,3> flux_abs_dif;
};


class Fluxes_byband_rt : public Fluxes_broadband_rt
{
    public:
        Fluxes_byband_rt(const int ncol_x, const int ncol_y, const int n_z, const int nlev, const int nbnd);
        virtual ~Fluxes_byband_rt() {};

        virtual void reduce(
                const Array_gpu<Float,3>& gpt_flux_up,
                const Array_gpu<Float,3>& gpt_flux_dn,
                const std::unique_ptr<Optical_props_arry_rt>& optical_props,
                const bool top_at_1);

        virtual void reduce(
                const Array_gpu<Float,3>& gpt_flux_up,
                const Array_gpu<Float,3>& gpt_flux_dn,
                const Array_gpu<Float,3>& gpt_flux_dn_dir,
                const std::unique_ptr<Optical_props_arry_rt>& optical_props,
                const bool top_at_1);

        Array_gpu<Float,3>& get_bnd_flux_up    () { return bnd_flux_up;     }
        Array_gpu<Float,3>& get_bnd_flux_dn    () { return bnd_flux_dn;     }
        Array_gpu<Float,3>& get_bnd_flux_dn_dir() { return bnd_flux_dn_dir; }
        Array_gpu<Float,3>& get_bnd_flux_net   () { return bnd_flux_net;    }

    private:
        Array_gpu<Float,3> bnd_flux_up;
        Array_gpu<Float,3> bnd_flux_dn;
        Array_gpu<Float,3> bnd_flux_dn_dir;
        Array_gpu<Float,3> bnd_flux_net;
};

//#endif
#endif
