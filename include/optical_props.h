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

#ifndef OPTICAL_PROPS_H
#define OPTICAL_PROPS_H

#include <memory>
#include "array.h"
#include "types.h"


class Optical_props
{
    public:
        Optical_props(
                const Array<Float,2>& band_lims_wvn,
                const Array<int,2>& band_lims_gpt);

        Optical_props(
                const Array<Float,2>& band_lims_wvn);

        virtual ~Optical_props() {};

        Optical_props(const Optical_props&) = default;

        const Array<int,1>& get_gpoint_bands() const { return this->gpt2band; }
        int get_nband() const { return this->band2gpt.dim(2); }
        int get_ngpt() const { return this->band2gpt.max(); }
        const Array<int,2>& get_band_lims_gpoint() const { return this->band2gpt; }
        const Array<Float,2>& get_band_lims_wavenumber() const { return this->band_lims_wvn; }

    private:
        Array<int,2> band2gpt;     // (begin g-point, end g-point) = band2gpt(2,band)
        Array<int,1> gpt2band;     // band = gpt2band(g-point)
        Array<Float,2> band_lims_wvn; // (upper and lower wavenumber by band) = band_lims_wvn(2,band)
};


// Base class for 1scl and 2str solvers fully implemented in header.
class Optical_props_arry : public Optical_props
{
    public:
        Optical_props_arry(const Optical_props& optical_props) :
            Optical_props(optical_props)
        {}

        virtual ~Optical_props_arry() {};

        virtual Array<Float,3>& get_tau() = 0;
        virtual Array<Float,3>& get_ssa() = 0;
        virtual Array<Float,3>& get_g  () = 0;

        virtual const Array<Float,3>& get_tau() const = 0;
        virtual const Array<Float,3>& get_ssa() const = 0;
        virtual const Array<Float,3>& get_g  () const = 0;

        // Optional argument.
        virtual void delta_scale(const Array<Float,3>& forward_frac=Array<Float,3>()) = 0;

        virtual void set_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e) = 0;

        virtual void get_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e) = 0;

        virtual int get_ncol() const = 0;
        virtual int get_nlay() const = 0;
};


class Optical_props_1scl : public Optical_props_arry
{
    public:
        // Initializer constructor.
        Optical_props_1scl(
                const int ncol,
                const int nlay,
                const Optical_props& optical_props);

        void set_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e);

        void get_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e);

        int get_ncol() const { return tau.dim(1); }
        int get_nlay() const { return tau.dim(2); }

        Array<Float,3>& get_tau() { return tau; }
        Array<Float,3>& get_ssa() { throw std::runtime_error("ssa is not available in this class"); }
        Array<Float,3>& get_g  () { throw std::runtime_error("g is available in this class"); }

        const Array<Float,3>& get_tau() const { return tau; }
        const Array<Float,3>& get_ssa() const { throw std::runtime_error("ssa is not available in this class"); }
        const Array<Float,3>& get_g  () const { throw std::runtime_error("g is not available in this class"); }

        void delta_scale(const Array<Float,3>& forward_frac=Array<Float,3>()) {}

    private:
        Array<Float,3> tau;
};


class Optical_props_2str : public Optical_props_arry
{
    public:
        Optical_props_2str(
                const int ncol,
                const int nlay,
                const Optical_props& optical_props);

        void set_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e);

        void get_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e);

        int get_ncol() const { return tau.dim(1); }
        int get_nlay() const { return tau.dim(2); }

        Array<Float,3>& get_tau() { return tau; }
        Array<Float,3>& get_ssa() { return ssa; }
        Array<Float,3>& get_g  () { return g; }

        const Array<Float,3>& get_tau() const { return tau; }
        const Array<Float,3>& get_ssa() const { return ssa; }
        const Array<Float,3>& get_g  () const { return g; }

        void delta_scale(const Array<Float,3>& forward_frac=Array<Float,3>());

    private:
        Array<Float,3> tau;
        Array<Float,3> ssa;
        Array<Float,3> g;
};


void add_to(Optical_props_1scl& op_inout, const Optical_props_1scl& op_in);
void add_to(Optical_props_2str& op_inout, const Optical_props_2str& op_in);


// GPU version of optical props class
#ifdef USECUDA

// Forward declare the classes in order to define add_to before the classes to enable friend function.
class Optical_props_1scl_gpu;
class Optical_props_2str_gpu;
void add_to(Optical_props_1scl_gpu& op_inout, const Optical_props_1scl_gpu& op_in);
void add_to(Optical_props_2str_gpu& op_inout, const Optical_props_2str_gpu& op_in);


class Optical_props_gpu
{
    public:
        Optical_props_gpu(
                const Array<Float,2>& band_lims_wvn,
                const Array<int,2>& band_lims_gpt);

        Optical_props_gpu(
                const Array<Float,2>& band_lims_wvn);

        virtual ~Optical_props_gpu() {};

        Optical_props_gpu(const Optical_props_gpu&) = default;

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
class Optical_props_arry_gpu : public Optical_props_gpu
{
    public:
        Optical_props_arry_gpu(const Optical_props_gpu& optical_props_gpu) :
            Optical_props_gpu(optical_props_gpu)
        {}

        virtual ~Optical_props_arry_gpu() {};

        virtual Array_gpu<Float,3>& get_tau() = 0;
        virtual Array_gpu<Float,3>& get_ssa() = 0;
        virtual Array_gpu<Float,3>& get_g  () = 0;

        virtual const Array_gpu<Float,3>& get_tau() const = 0;
        virtual const Array_gpu<Float,3>& get_ssa() const = 0;
        virtual const Array_gpu<Float,3>& get_g  () const = 0;

        // Optional argument.
        virtual void delta_scale(const Array_gpu<Float,3>& forward_frac=Array_gpu<Float,3>()) = 0;

        virtual int get_ncol() const = 0;
        virtual int get_nlay() const = 0;
};


class Optical_props_1scl_gpu : public Optical_props_arry_gpu
{
    public:
        // Initializer constructor.
        Optical_props_1scl_gpu(
                const int ncol,
                const int nlay,
                const Optical_props_gpu& optical_props_gpu);

        int get_ncol() const { return tau.dim(1); }
        int get_nlay() const { return tau.dim(2); }

        Array_gpu<Float,3>& get_tau() { return tau; }
        Array_gpu<Float,3>& get_ssa() { throw std::runtime_error("ssa is not available in this class"); }
        Array_gpu<Float,3>& get_g  () { throw std::runtime_error("g is available in this class"); }

        const Array_gpu<Float,3>& get_tau() const { return tau; }
        const Array_gpu<Float,3>& get_ssa() const { throw std::runtime_error("ssa is not available in this class"); }
        const Array_gpu<Float,3>& get_g  () const { throw std::runtime_error("g is available in this class"); }

        void delta_scale(const Array_gpu<Float,3>& forward_frac=Array_gpu<Float,3>()) {}

    private:
        Array_gpu<Float,3> tau;
};


class Optical_props_2str_gpu : public Optical_props_arry_gpu
{
    public:
        Optical_props_2str_gpu(
                const int ncol,
                const int nlay,
                const Optical_props_gpu& optical_props_gpu);

        int get_ncol() const { return tau.dim(1); }
        int get_nlay() const { return tau.dim(2); }

        Array_gpu<Float,3>& get_tau() { return tau; }
        Array_gpu<Float,3>& get_ssa() { return ssa; }
        Array_gpu<Float,3>& get_g  () { return g; }

        const Array_gpu<Float,3>& get_tau() const { return tau; }
        const Array_gpu<Float,3>& get_ssa() const { return ssa; }
        const Array_gpu<Float,3>& get_g  () const { return g; }

        void delta_scale(const Array_gpu<Float,3>& forward_frac=Array_gpu<Float,3>());

    private:
        Array_gpu<Float,3> tau;
        Array_gpu<Float,3> ssa;
        Array_gpu<Float,3> g;

        friend void add_to(Optical_props_2str_gpu& op_inout, const Optical_props_2str_gpu& op_in);
};
#endif

#endif
