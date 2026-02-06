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

#ifndef OPTICAL_PROPS_RT_H
#define OPTICAL_PROPS_RT_H


#include <memory>
#include "array.h"
#include "types.h"

// GPU version of optical props class
#ifdef USECUDA

// Forward declare the classes in order to define add_to before the classes to enable friend function.
class Optical_props_1scl_rt;
class Optical_props_2str_rt;
void add_to(Optical_props_1scl_rt& op_inout, const Optical_props_1scl_rt& op_in);
void add_to(Optical_props_2str_rt& op_inout, const Optical_props_2str_rt& op_in);


class Optical_props_rt
{
    public:
        Optical_props_rt(
                const Array<Float,2>& band_lims_wvn,
                const Array<int,2>& band_lims_gpt);

        Optical_props_rt(
                const Array<Float,2>& band_lims_wvn);

        virtual ~Optical_props_rt() {};

        Optical_props_rt(const Optical_props_rt&) = default;

        Array<int,1> get_gpoint_bands() const { return this->gpt2band; }
 
        const Array_gpu<int,1>& get_gpoint_bands_gpu() const { return this->gpt2band_gpu; }
        const Array_gpu<int,2>& get_band_lims_gpoint_gpu() const { return this->band2gpt_gpu;}
        int get_nband() const { return this->band2gpt.dim(2); }
        int get_ngpt() const { return this->band2gpt.max(); }
        const Array<int,2>& get_band_lims_gpoint() const { return this->band2gpt;}
        const Array<Float,2>& get_band_lims_wavenumber() const { return this->band_lims_wvn; }
        
    private:
        Array<int,2> band2gpt;         // (begin g-point, end g-point) = band2gpt(2,band)
        Array_gpu<int,2> band2gpt_gpu; // (begin g-point, end g-point) = band2gpt(2,band)
        
        Array<int,1> gpt2band;         // band = gpt2band(g-point)
        Array_gpu<int,1> gpt2band_gpu; // band = gpt2band(g-point)
        
        Array<Float,2> band_lims_wvn; // (upper and lower wavenumber by band) = band_lims_wvn(2,band)
};

// Base class for 1scl and 2str solvers fully implemented in header.
class Optical_props_arry_rt : public Optical_props_rt
{
    public:
        Optical_props_arry_rt(const Optical_props_rt& optical_props_rt) :
            Optical_props_rt(optical_props_rt)
        {}

        virtual ~Optical_props_arry_rt() {};

        virtual Array_gpu<Float,2>& get_tau() = 0;
        virtual Array_gpu<Float,2>& get_ssa() = 0;
        virtual Array_gpu<Float,2>& get_g  () = 0;

        virtual const Array_gpu<Float,2>& get_tau() const = 0;
        virtual const Array_gpu<Float,2>& get_ssa() const = 0;
        virtual const Array_gpu<Float,2>& get_g  () const = 0;

        // Optional argument.
        virtual void delta_scale(const Array_gpu<Float,3>& forward_frac=Array_gpu<Float,3>()) = 0;

        virtual int get_ncol() const = 0;
        virtual int get_nlay() const = 0;
};

class Optical_props_1scl_rt : public Optical_props_arry_rt
{
    public:
        // Initializer constructor.
        Optical_props_1scl_rt(
                const int ncol,
                const int nlay,
                const Optical_props_rt& optical_props_rt);

        int get_ncol() const { return tau.dim(1); }
        int get_nlay() const { return tau.dim(2); }

        Array_gpu<Float,2>& get_tau() { return tau; }
        Array_gpu<Float,2>& get_ssa() { throw std::runtime_error("ssa is not available in this class"); }
        Array_gpu<Float,2>& get_g  () { throw std::runtime_error("g is available in this class"); }

        const Array_gpu<Float,2>& get_tau() const { return tau; }
        const Array_gpu<Float,2>& get_ssa() const { throw std::runtime_error("ssa is not available in this class"); }
        const Array_gpu<Float,2>& get_g  () const { throw std::runtime_error("g is available in this class"); }

        void delta_scale(const Array_gpu<Float,3>& forward_frac=Array_gpu<Float,3>()) {}

    private:
        Array_gpu<Float,2> tau;
};

class Optical_props_2str_rt : public Optical_props_arry_rt
{
    public:
        Optical_props_2str_rt(
                const int ncol,
                const int nlay,
                const Optical_props_rt& optical_props_rt);

        int get_ncol() const { return tau.dim(1); }
        int get_nlay() const { return tau.dim(2); }

        Array_gpu<Float,2>& get_tau() { return tau; }
        Array_gpu<Float,2>& get_ssa() { return ssa; }
        Array_gpu<Float,2>& get_g  () { return g; }

        const Array_gpu<Float,2>& get_tau() const { return tau; }
        const Array_gpu<Float,2>& get_ssa() const { return ssa; }
        const Array_gpu<Float,2>& get_g  () const { return g; }

        void delta_scale(const Array_gpu<Float,3>& forward_frac=Array_gpu<Float,3>());

    private:
        Array_gpu<Float,2> tau;
        Array_gpu<Float,2> ssa;
        Array_gpu<Float,2> g;

        friend void add_to(Optical_props_2str_rt& op_inout, const Optical_props_2str_rt& op_in);
};
#endif

#endif
