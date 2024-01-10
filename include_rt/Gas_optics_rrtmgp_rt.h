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

#ifndef GAS_OPTICS_RRTMGP_RT_H
#define GAS_OPTICS_RRTMGP_RT_H

#include <string>

#include "Array.h"
#include "Gas_optics_rt.h"
#include "types.h"

// Forward declarations.
// template<typename Float> class Gas_optics;
class Optical_props_rt;
class Optical_props_arry_rt;
class Gas_concs_gpu;
class Source_func_lw_rt;

#ifdef USECUDA
class Gas_optics_rrtmgp_rt : public Gas_optics_rt
{
    public:
        // Constructor for longwave variant.
        Gas_optics_rrtmgp_rt(
                const Gas_concs_gpu& available_gases,
                const Array<std::string,1>& gas_names,
                const Array<int,3>& key_species,
                const Array<int,2>& band2gpt,
                const Array<Float,2>& band_lims_wavenum,
                const Array<Float,1>& press_ref,
                const Float press_ref_trop,
                const Array<Float,1>& temp_ref,
                const Float temp_ref_p,
                const Float temp_ref_t,
                const Array<Float,3>& vmr_ref,
                const Array<Float,4>& kmajor,
                const Array<Float,3>& kminor_lower,
                const Array<Float,3>& kminor_upper,
                const Array<std::string,1>& gas_minor,
                const Array<std::string,1>& identifier_minor,
                const Array<std::string,1>& minor_gases_lower,
                const Array<std::string,1>& minor_gases_upper,
                const Array<int,2>& minor_limits_gpt_lower,
                const Array<int,2>& minor_limits_gpt_upper,
                const Array<Bool,1>& minor_scales_with_density_lower,
                const Array<Bool,1>& minor_scales_with_density_upper,
                const Array<std::string,1>& scaling_gas_lower,
                const Array<std::string,1>& scaling_gas_upper,
                const Array<Bool,1>& scale_by_complement_lower,
                const Array<Bool,1>& scale_by_complement_upper,
                const Array<int,1>& kminor_start_lower,
                const Array<int,1>& kminor_start_upper,
                const Array<Float,2>& totplnk,
                const Array<Float,4>& planck_frac,
                const Array<Float,3>& rayl_lower,
                const Array<Float,3>& rayl_upper);

        // Constructor for shortwave variant.
        Gas_optics_rrtmgp_rt(
                const Gas_concs_gpu& available_gases,
                const Array<std::string,1>& gas_names,
                const Array<int,3>& key_species,
                const Array<int,2>& band2gpt,
                const Array<Float,2>& band_lims_wavenum,
                const Array<Float,1>& press_ref,
                const Float press_ref_trop,
                const Array<Float,1>& temp_ref,
                const Float temp_ref_p,
                const Float temp_ref_t,
                const Array<Float,3>& vmr_ref,
                const Array<Float,4>& kmajor,
                const Array<Float,3>& kminor_lower,
                const Array<Float,3>& kminor_upper,
                const Array<std::string,1>& gas_minor,
                const Array<std::string,1>& identifier_minor,
                const Array<std::string,1>& minor_gases_lower,
                const Array<std::string,1>& minor_gases_upper,
                const Array<int,2>& minor_limits_gpt_lower,
                const Array<int,2>& minor_limits_gpt_upper,
                const Array<Bool,1>& minor_scales_with_density_lower,
                const Array<Bool,1>& minor_scales_with_density_upper,
                const Array<std::string,1>& scaling_gas_lower,
                const Array<std::string,1>& scaling_gas_upper,
                const Array<Bool,1>& scale_by_complement_lower,
                const Array<Bool,1>& scale_by_complement_upper,
                const Array<int,1>& kminor_start_lower,
                const Array<int,1>& kminor_start_upper,
                const Array<Float,1>& solar_src_quiet,
                const Array<Float,1>& solar_src_facular,
                const Array<Float,1>& solar_src_sunspot,
                const Float tsi_default,
                const Float mg_default,
                const Float sb_default,
                const Array<Float,3>& rayl_lower,
                const Array<Float,3>& rayl_upper);

        static void get_col_dry(
                Array_gpu<Float,2>& col_dry,
                const Array_gpu<Float,2>& vmr_h2o,
                const Array_gpu<Float,2>& plev);

        bool source_is_internal() const { return (totplnk.size() > 0) && (planck_frac.size() > 0); }
        bool source_is_external() const { return (solar_source.size() > 0); }

        Float get_press_ref_min() const { return press_ref_min; }
        Float get_press_ref_max() const { return press_ref_max; }

        Float get_temp_min() const { return temp_ref_min; }
        Float get_temp_max() const { return temp_ref_max; }

        int get_nflav() const { return flavor.dim(2); }
        int get_neta() const { return kmajor.dim(2); }
        int get_npres() const { return kmajor.dim(3)-1; }
        int get_ntemp() const { return kmajor.dim(1); }
        int get_nPlanckTemp() const { return totplnk.dim(1); }

        Float get_tsi() const;
        Float band_source(const int gpt_start, const int gpt_end) const;

        // Longwave variant.
        void gas_optics(
                const int igpt, const int col_s, const int ncol_block, const int ncol,
                const Array_gpu<Float,2>& play,
                const Array_gpu<Float,2>& plev,
                const Array_gpu<Float,2>& tlay,
                const Array_gpu<Float,1>& tsfc,
                const Gas_concs_gpu& gas_desc,
                std::unique_ptr<Optical_props_arry_rt>& optical_props,
                Source_func_lw_rt& sources,
                const Array_gpu<Float,2>& col_dry,
                const Array_gpu<Float,2>& tlev);

        // shortwave variant
        void gas_optics(
                const int igpt, const int col_s, const int ncol_block, const int ncol,
                const Array_gpu<Float,2>& play,
                const Array_gpu<Float,2>& plev,
                const Array_gpu<Float,2>& tlay,
                const Gas_concs_gpu& gas_desc,
                std::unique_ptr<Optical_props_arry_rt>& optical_props,
                Array_gpu<Float,1>& toa_src,
                const Array_gpu<Float,2>& col_dry);

    private:
        Array<Float,2> totplnk;
        Array<Float,4> planck_frac;
        Float totplnk_delta;
        Float temp_ref_min, temp_ref_max;
        Float press_ref_min, press_ref_max;
        Float press_ref_trop_log;

        Float press_ref_log_delta;
        Float temp_ref_delta;

        Array<Float,1> press_ref, press_ref_log, temp_ref;

        Array<std::string,1> gas_names;

        Array<Float,3> vmr_ref;

        Array<int,2> flavor;
        Array<int,2> gpoint_flavor;

        Array<Float,4> kmajor;

        Array<Float,3> kminor_lower;
        Array<Float,3> kminor_upper;

        Array<int,2> minor_limits_gpt_lower;
        Array<int,2> minor_limits_gpt_upper;

        Array<int,2> first_last_minor_lower;
        Array<int,2> first_last_minor_upper;

        Array<Bool,1> minor_scales_with_density_lower;
        Array<Bool,1> minor_scales_with_density_upper;

        Array<Bool,1> scale_by_complement_lower;
        Array<Bool,1> scale_by_complement_upper;

        Array<int,1> kminor_start_lower;
        Array<int,1> kminor_start_upper;

        Array<int,1> idx_minor_lower;
        Array<int,1> idx_minor_upper;

        Array<int,1> idx_minor_scaling_lower;
        Array<int,1> idx_minor_scaling_upper;

        Array<int,1> is_key;

        Array<Float,1> solar_source_quiet;
        Array<Float,1> solar_source_facular;
        Array<Float,1> solar_source_sunspot;
        Array<Float,1> solar_source;

        Array<Float,4> krayl;

        int idx_h2o;
        
        Array_gpu<Float,1> solar_source_g;
        Array_gpu<Float,2> totplnk_gpu;
        Array_gpu<Float,4> planck_frac_gpu;
        Array_gpu<Float,1> press_ref_gpu, press_ref_log_gpu, temp_ref_gpu;
        Array_gpu<Float,3> vmr_ref_gpu;
        Array_gpu<int,2> flavor_gpu;
        Array_gpu<int,2> gpoint_flavor_gpu;
        Array_gpu<Float,4> kmajor_gpu;
        Array_gpu<Float,3> kminor_lower_gpu;
        Array_gpu<Float,3> kminor_upper_gpu;
        Array_gpu<int,2> minor_limits_gpt_lower_gpu;
        Array_gpu<int,2> minor_limits_gpt_upper_gpu;
        Array_gpu<int,2> first_last_minor_lower_gpu;
        Array_gpu<int,2> first_last_minor_upper_gpu;
        Array_gpu<Bool,1> minor_scales_with_density_lower_gpu;
        Array_gpu<Bool,1> minor_scales_with_density_upper_gpu;
        Array_gpu<Bool,1> scale_by_complement_lower_gpu;
        Array_gpu<Bool,1> scale_by_complement_upper_gpu;
        Array_gpu<int,1> kminor_start_lower_gpu;
        Array_gpu<int,1> kminor_start_upper_gpu;
        Array_gpu<int,1> idx_minor_lower_gpu;
        Array_gpu<int,1> idx_minor_upper_gpu;
        Array_gpu<int,1> idx_minor_scaling_lower_gpu;
        Array_gpu<int,1> idx_minor_scaling_upper_gpu;
        Array_gpu<Float,1> solar_source_gpu;
        Array_gpu<Float,4> krayl_gpu;
        Array_gpu<int,2> jtemp;
        Array_gpu<int,2> jpress;;
        Array_gpu<Bool,2> tropo;
        Array_gpu<Float,5> fmajor;
        Array_gpu<int,3> jeta;
        Array_gpu<Float,3> vmr;
        Array_gpu<Float,3> col_gas;
        Array_gpu<Float,3> col_mix;
        Array_gpu<Float,4> fminor;

        int get_ngas() const { return this->gas_names.dim(1); }

        void init_abs_coeffs(
                const Gas_concs_gpu& available_gases,
                const Array<std::string,1>& gas_names,
                const Array<int,3>& key_species,
                const Array<int,2>& band2gpt,
                const Array<Float,2>& band_lims_wavenum,
                const Array<Float,1>& press_ref,
                const Array<Float,1>& temp_ref,
                const Float press_ref_trop,
                const Float temp_ref_p,
                const Float temp_ref_t,
                const Array<Float,3>& vmr_ref,
                const Array<Float,4>& kmajor,
                const Array<Float,3>& kminor_lower,
                const Array<Float,3>& kminor_upper,
                const Array<std::string,1>& gas_minor,
                const Array<std::string,1>& identifier_minor,
                const Array<std::string,1>& minor_gases_lower,
                const Array<std::string,1>& minor_gases_upper,
                const Array<int,2>& minor_limits_gpt_lower,
                const Array<int,2>& minor_limits_gpt_upper,
                const Array<Bool,1>& minor_scales_with_density_lower,
                const Array<Bool,1>& minor_scales_with_density_upper,
                const Array<std::string,1>& scaling_gas_lower,
                const Array<std::string,1>& scaling_gas_upper,
                const Array<Bool,1>& scale_by_complement_lower,
                const Array<Bool,1>& scale_by_complement_upper,
                const Array<int,1>& kminor_start_lower,
                const Array<int,1>& kminor_start_upper,
                const Array<Float,3>& rayl_lower,
                const Array<Float,3>& rayl_upper);

        void set_solar_variability(
                const Float md_index, const Float sb_index);

        void compute_gas_taus(
                const int col_s, const int ncol_block, const int ncol, const int nlay, 
                const int ngpt, const int nband, const int igpt,
                const Array_gpu<Float,2>& play,
                const Array_gpu<Float,2>& plev,
                const Array_gpu<Float,2>& tlay,
                const Gas_concs_gpu& gas_desc,
                std::unique_ptr<Optical_props_arry_rt>& optical_props,
                Array_gpu<int,2>& jtemp, Array_gpu<int,2>& jpress,
                Array_gpu<int,3>& jeta,
                Array_gpu<Bool,2>& tropo,
                Array_gpu<Float,5>& fmajor,
                const Array_gpu<Float,2>& col_dry);

        void combine_abs_and_rayleigh(
                const Array_gpu<Float,2>& tau,
                const Array_gpu<Float,2>& tau_rayleigh,
                std::unique_ptr<Optical_props_arry_rt>& optical_props);

        void source(
                const int ncol, const int nlay, const int nband, const int ngpt, const int igpt,
                const Array_gpu<Float,2>& play, const Array_gpu<Float,2>& plev,
                const Array_gpu<Float,2>& tlay, const Array_gpu<Float,1>& tsfc,
                const Array_gpu<int,2>& jtemp, const Array_gpu<int,2>& jpress,
                const Array_gpu<int,3>& jeta, const Array_gpu<Bool,2>& tropo,
                const Array_gpu<Float,5>& fmajor,
                Source_func_lw_rt& sources,
                const Array_gpu<Float,2>& tlev);

        void find_relevant_gases_gpt(
            const int igpt,
            std::vector<int>& gases);
};
#endif

#endif
