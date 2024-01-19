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
#include <chrono>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <boost/algorithm/string.hpp>

#include "Optical_props_rt.h"
#include "Source_functions_rt.h"

#include "Gas_concs.h"
#include "Gas_optics_rrtmgp_rt.h"
#include "Array.h"
#include "iostream"

#include "gas_optics_rrtmgp_kernels_cuda_rt.h"


namespace
{
    __global__
    void spread_col_kernel(
            const int ncol, const int igpt, Float* __restrict__ src_out, const Float* __restrict__ src_in)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;

        if (icol < ncol)
        {
            src_out[icol] = src_in[igpt];
        }
    }


    void spread_col(
            const int ncol, const int igpt, Array_gpu<Float,1>& src_out, const Array_gpu<Float,1>& src_in)
    {
        const int block_col = 16;
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, 1);
        dim3 block_gpu(block_col, 1);

        spread_col_kernel<<<grid_gpu, block_gpu>>>(
            ncol, igpt-1, src_out.ptr(), src_in.ptr());
    }


    int find_index(
            const Array<std::string,1>& data, const std::string& value)
    {
        auto it = std::find(data.v().begin(), data.v().end(), value);
        if (it == data.v().end())
            return -1;
        else
            return it - data.v().begin() + 1;
    }

    void add_if_not_in_vector(
            std::vector<int>& vec, const int value)
    {
        if(!(std::find(vec.begin(), vec.end(), value) != vec.end()))
            vec.push_back(value);

    }


    void reduce_minor_arrays(
                const Gas_concs_gpu& available_gases,
                const Array<std::string,1>& gas_names,
                const Array<std::string,1>& gas_minor,
                const Array<std::string,1>& identifier_minor,
                const Array<Float,3>& kminor_atm,
                const Array<std::string,1>& minor_gases_atm,
                const Array<int,2>& minor_limits_gpt_atm,
                const Array<Bool,1>& minor_scales_with_density_atm,
                const Array<std::string,1>& scaling_gas_atm,
                const Array<Bool,1>& scale_by_complement_atm,
                const Array<int,1>& kminor_start_atm,

                Array<Float,3>& kminor_atm_red,
                Array<std::string,1>& minor_gases_atm_red,
                Array<int,2>& minor_limits_gpt_atm_red,
                Array<Bool,1>& minor_scales_with_density_atm_red,
                Array<std::string,1>& scaling_gas_atm_red,
                Array<Bool,1>& scale_by_complement_atm_red,
                Array<int,1>& kminor_start_atm_red)
    {
        int nm = minor_gases_atm.dim(1);
        int tot_g = 0;

        Array<Bool,1> gas_is_present({nm});

        for (int i=1; i<=nm; ++i)
        {
            const int idx_mnr = find_index(identifier_minor, minor_gases_atm({i}));

            // Search for
            std::string gas_minor_trimmed = gas_minor({idx_mnr});
            boost::trim(gas_minor_trimmed);

            gas_is_present({i}) = available_gases.exists(gas_minor_trimmed);
            if (gas_is_present({i}))
                tot_g += minor_limits_gpt_atm({2,i}) - minor_limits_gpt_atm({1,i}) + 1;
        }

        const int red_nm = std::accumulate(gas_is_present.v().begin(), gas_is_present.v().end(), 0);

        Array<Float,3> kminor_atm_red_t;

        if (red_nm == nm)
        {
            kminor_atm_red_t = kminor_atm;
            minor_gases_atm_red = minor_gases_atm;
            minor_limits_gpt_atm_red = minor_limits_gpt_atm;
            minor_scales_with_density_atm_red = minor_scales_with_density_atm;
            scaling_gas_atm_red = scaling_gas_atm;
            scale_by_complement_atm_red = scale_by_complement_atm;
            kminor_start_atm_red = kminor_start_atm;
        }
        else
        {
            // Use a lambda function as the operation has to be repeated many times.
            auto resize_and_set = [&](auto& a_red, const auto& a)
            {
                a_red.set_dims({red_nm});
                int counter = 1;
                for (int i=1; i<=gas_is_present.dim(1); ++i)
                {
                    if (gas_is_present({i}))
                    {
                       a_red({counter}) = a({i});
                       ++counter;
                    }
                }
            };

            resize_and_set(minor_gases_atm_red, minor_gases_atm);
            resize_and_set(minor_scales_with_density_atm_red, minor_scales_with_density_atm);
            resize_and_set(scaling_gas_atm_red, scaling_gas_atm);
            resize_and_set(scale_by_complement_atm_red, scale_by_complement_atm);
            resize_and_set(kminor_start_atm_red, kminor_start_atm);

            minor_limits_gpt_atm_red.set_dims({2, red_nm});
            kminor_atm_red_t.set_dims({tot_g, kminor_atm.dim(2), kminor_atm.dim(3)});

            int icnt = 0;
            int n_elim = 0;
            for (int i=1; i<=nm; ++i)
            {
                int ng = minor_limits_gpt_atm({2,i}) - minor_limits_gpt_atm({1,i}) + 1;
                if (gas_is_present({i}))
                {
                    ++icnt;
                    minor_limits_gpt_atm_red({1,icnt}) = minor_limits_gpt_atm({1,i});
                    minor_limits_gpt_atm_red({2,icnt}) = minor_limits_gpt_atm({2,i});
                    kminor_start_atm_red({icnt}) = kminor_start_atm({i}) - n_elim;

                    for (int j=1; j<=ng; ++j)
                        for (int i2=1; i2<=kminor_atm.dim(2); ++i2)
                            for (int i3=1; i3<=kminor_atm.dim(3); ++i3)
                                kminor_atm_red_t({kminor_start_atm_red({icnt})+j-1,i2,i3}) =
                                        kminor_atm({kminor_start_atm({i})+j-1,i2,i3});
                }
                else
                    n_elim += ng;
            }
        }

        // Reshape following the new ordering in v1.5.
        kminor_atm_red.set_dims({kminor_atm_red_t.dim(3), kminor_atm_red_t.dim(2), kminor_atm_red_t.dim(1)});
        for (int i3=1; i3<=kminor_atm_red.dim(3); ++i3)
            for (int i2=1; i2<=kminor_atm_red.dim(2); ++i2)
                for (int i1=1; i1<=kminor_atm_red.dim(1); ++i1)
                    kminor_atm_red({i1, i2, i3}) = kminor_atm_red_t({i3, i2, i1});
    }

    void create_idx_minor(
            const Array<std::string,1>& gas_names,
            const Array<std::string,1>& gas_minor,
            const Array<std::string,1>& identifier_minor,
            const Array<std::string,1>& minor_gases_atm,
            Array<int,1>& idx_minor_atm)
    {
        Array<int,1> idx_minor_atm_out({minor_gases_atm.dim(1)});

        for (int imnr=1; imnr<=minor_gases_atm.dim(1); ++imnr)
        {
            // Find identifying string for minor species in list of possible identifiers (e.g. h2o_slf)
            const int idx_mnr = find_index(identifier_minor, minor_gases_atm({imnr}));

            // Find name of gas associated with minor species identifier (e.g. h2o)
            idx_minor_atm_out({imnr}) = find_index(gas_names, gas_minor({idx_mnr}));
        }

        idx_minor_atm = idx_minor_atm_out;
    }

    void create_idx_minor_scaling(
            const Array<std::string,1>& gas_names,
            const Array<std::string,1>& scaling_gas_atm,
            Array<int,1>& idx_minor_scaling_atm)
    {
        Array<int,1> idx_minor_scaling_atm_out({scaling_gas_atm.dim(1)});

        for (int imnr=1; imnr<=scaling_gas_atm.dim(1); ++imnr)
            idx_minor_scaling_atm_out({imnr}) = find_index(gas_names, scaling_gas_atm({imnr}));

        idx_minor_scaling_atm = idx_minor_scaling_atm_out;
    }

    void create_key_species_reduce(
            const Array<std::string,1>& gas_names,
            const Array<std::string,1>& gas_names_red,
            const Array<int,3>& key_species,
            Array<int,3>& key_species_red,
            Array<Bool,1>& key_species_present_init)
    {
        const int np = key_species.dim(1);
        const int na = key_species.dim(2);
        const int nt = key_species.dim(3);

        key_species_red.set_dims({key_species.dim(1), key_species.dim(2), key_species.dim(3)});
        key_species_present_init.set_dims({gas_names.dim(1)});

        for (int i=1; i<=key_species_present_init.dim(1); ++i)
            key_species_present_init({i}) = 1;

        for (int ip=1; ip<=np; ++ip)
            for (int ia=1; ia<=na; ++ia)
                for (int it=1; it<=nt; ++it)
                {
                    const int ks = key_species({ip,ia,it});
                    if (ks != 0)
                    {
                        const int ksr = find_index(gas_names_red, gas_names({ks}));
                        key_species_red({ip,ia,it}) = ksr;
                        if (ksr == -1)
                            key_species_present_init({ks}) = 0;
                    }
                    else
                        key_species_red({ip,ia,it}) = ks;
                }
    }

    void check_key_species_present_init(
            const Array<std::string,1>& gas_names,
            const Array<Bool,1>& key_species_present_init
            )
    {
        for (int i=1; i<=key_species_present_init.dim(1); ++i)
        {
            if (key_species_present_init({i}) == 0)
            {
                std::string error_message = "Gas optics: required gas " + gas_names({i}) + " is missing";
                throw std::runtime_error(error_message);
            }
        }
    }

    void create_flavor(
            const Array<int,3>& key_species,
            Array<int,2>& flavor)
    {
        Array<int,2> key_species_list({2, key_species.dim(3)*2});

        // Prepare list of key species.
        int i = 1;
        for (int ibnd=1; ibnd<=key_species.dim(3); ++ibnd)
            for (int iatm=1; iatm<=key_species.dim(2); ++iatm)
            {
                key_species_list({1,i}) = key_species({1,iatm,ibnd});
                key_species_list({2,i}) = key_species({2,iatm,ibnd});
                ++i;
            }

        // Rewrite single key_species pairs.
        for (int i=1; i<=key_species_list.dim(2); ++i)
        {
            if ( key_species_list({1,i}) == 0 && key_species_list({2,i}) == 0 )
            {
                key_species_list({1,i}) = 2;
                key_species_list({2,i}) = 2;
            }
        }

        // Count unique key species pairs.
        int iflavor = 0;
        for (int i=1; i<=key_species_list.dim(2); ++i)
        {
            bool pair_exists = false;
            for (int ii=1; ii<=i-1; ++ii)
            {
                if ( (key_species_list({1,i}) == key_species_list({1,ii})) &&
                     (key_species_list({2,i}) == key_species_list({2,ii})) )
                {
                    pair_exists = true;
                    break;
                }
            }
            if (!pair_exists)
                ++iflavor;
        }

        // Fill flavors.
        flavor.set_dims({2,iflavor});
        iflavor = 0;
        for (int i=1; i<=key_species_list.dim(2); ++i)
        {
            bool pair_exists = false;
            for (int ii=1; ii<=i-1; ++ii)
            {
                if ( (key_species_list({1,i}) == key_species_list({1,ii})) &&
                     (key_species_list({2,i}) == key_species_list({2,ii})) )
                {
                    pair_exists = true;
                    break;
                }
            }
            if (!pair_exists)
            {
                ++iflavor;
                flavor({1,iflavor}) = key_species_list({1,i});
                flavor({2,iflavor}) = key_species_list({2,i});
            }
        }
    }

    int key_species_pair2flavor(
            const Array<int,2>& flavor,
            const Array<int,1>& key_species_pair)
    {
        // Search for match.
        for (int iflav=1; iflav<=flavor.dim(2); ++iflav)
        {
            if ( key_species_pair({1}) == flavor({1, iflav}) &&
                 key_species_pair({2}) == flavor({2, iflav}) )
                return iflav;
        }

        // No match found.
        return -1;
    }

    void create_gpoint_flavor(
            const Array<int,3>& key_species,
            const Array<int,1>& gpt2band,
            const Array<int,2>& flavor,
            Array<int,2>& gpoint_flavor)
    {
        const int ngpt = gpt2band.dim(1);
        gpoint_flavor.set_dims({2,ngpt});

        for (int igpt=1; igpt<=ngpt; ++igpt)
            for (int iatm=1; iatm<=2; ++iatm)
            {
                int pair_1 = key_species( {1, iatm, gpt2band({igpt})} );
                int pair_2 = key_species( {2, iatm, gpt2band({igpt})} );

                // Rewrite species pair.
                Array<int,1> rewritten_pair({2});
                if (pair_1 == 0 && pair_2 == 0)
                {
                    rewritten_pair({1}) = 2;
                    rewritten_pair({2}) = 2;
                }
                else
                {
                    rewritten_pair({1}) = pair_1;
                    rewritten_pair({2}) = pair_2;
                }

                // Write the output.
                gpoint_flavor({iatm,igpt}) = key_species_pair2flavor(
                        flavor, rewritten_pair);
            }
    }

    // which minor indices belong to each g-point
    void invert_minor_gpt_limits(
        const Array<int,2>& minor_limits_gpt,
        Array<int,2>& first_last_minor)
    {
        first_last_minor.fill(-1);
        for (int i1=1; i1<=minor_limits_gpt.dim(2); ++i1)
        {
            for (int i2=minor_limits_gpt({1,i1}); i2<=minor_limits_gpt({2,i1}); ++i2)
            {
                if (first_last_minor({1,i2}) == -1)
                {
                    first_last_minor({1,i2}) = i1-1;
                }
                first_last_minor({2,i2}) = i1-1;
            }

        }
    }

}


// Constructor of longwave variant.

Gas_optics_rrtmgp_rt::Gas_optics_rrtmgp_rt(
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
        const Array<Float,3>& rayl_upper) :
            Gas_optics_rt(band_lims_wavenum, band2gpt),
            totplnk(totplnk)
{
    // Reshaping according to new dimension ordering since v1.5
    this->planck_frac.set_dims({planck_frac.dim(4), planck_frac.dim(2), planck_frac.dim(3), planck_frac.dim(1)});
    for (int i4=1; i4<=this->planck_frac.dim(4); ++i4)
        for (int i3=1; i3<=this->planck_frac.dim(3); ++i3)
            for (int i2=1; i2<=this->planck_frac.dim(2); ++i2)
                for (int i1=1; i1<=this->planck_frac.dim(1); ++i1)
                    this->planck_frac({i1, i2, i3, i4}) = planck_frac({i4, i2, i3, i1});

    // Initialize the absorption coefficient array, including Rayleigh scattering
    // tables if provided.
    init_abs_coeffs(
            available_gases,
            gas_names, key_species,
            band2gpt, band_lims_wavenum,
            press_ref, temp_ref,
            press_ref_trop, temp_ref_p, temp_ref_t,
            vmr_ref,
            kmajor, kminor_lower, kminor_upper,
            gas_minor,identifier_minor,
            minor_gases_lower, minor_gases_upper,
            minor_limits_gpt_lower,
            minor_limits_gpt_upper,
            minor_scales_with_density_lower,
            minor_scales_with_density_upper,
            scaling_gas_lower, scaling_gas_upper,
            scale_by_complement_lower,
            scale_by_complement_upper,
            kminor_start_lower,
            kminor_start_upper,
            rayl_lower, rayl_upper);

    // Temperature steps for Planck function interpolation.
    // Assumes that temperature minimum and max are the same for the absorption coefficient grid and the
    // Planck grid and the Planck grid is equally spaced.
    totplnk_delta = (temp_ref_max - temp_ref_min) / (totplnk.dim(1)-1);
}


// Constructor of the shortwave variant.

Gas_optics_rrtmgp_rt::Gas_optics_rrtmgp_rt(
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
        const Array<Float,1>& solar_source_quiet,
        const Array<Float,1>& solar_source_facular,
        const Array<Float,1>& solar_source_sunspot,
        const Float tsi_default,
        const Float mg_default,
        const Float sb_default,
        const Array<Float,3>& rayl_lower,
        const Array<Float,3>& rayl_upper) :
            Gas_optics_rt(band_lims_wavenum, band2gpt)
{
    // Initialize the absorption coefficient array, including Rayleigh scattering
    // tables if provided.
    init_abs_coeffs(
            available_gases,
            gas_names, key_species,
            band2gpt, band_lims_wavenum,
            press_ref, temp_ref,
            press_ref_trop, temp_ref_p, temp_ref_t,
            vmr_ref,
            kmajor, kminor_lower, kminor_upper,
            gas_minor,identifier_minor,
            minor_gases_lower, minor_gases_upper,
            minor_limits_gpt_lower,
            minor_limits_gpt_upper,
            minor_scales_with_density_lower,
            minor_scales_with_density_upper,
            scaling_gas_lower, scaling_gas_upper,
            scale_by_complement_lower,
            scale_by_complement_upper,
            kminor_start_lower,
            kminor_start_upper,
            rayl_lower, rayl_upper);

    // Compute the solar source.
    this->solar_source_quiet = solar_source_quiet;
    this->solar_source_facular = solar_source_facular;
    this->solar_source_sunspot = solar_source_sunspot;

    this->solar_source.set_dims(solar_source_quiet.get_dims());

    set_solar_variability(mg_default, sb_default);
}



void Gas_optics_rrtmgp_rt::init_abs_coeffs(
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
        const Array<Float,3>& rayl_upper)
{
    // Which gases known to the gas optics are present in the host model (available_gases)?
    std::vector<std::string> gas_names_to_use;

    for (const std::string &s : gas_names.v())
    {
        if (available_gases.exists(s))
            gas_names_to_use.push_back(s);
    }

    // Now the number of gases is the union of those known to the k-distribution and provided
    // by the host model.
    const int n_gas = gas_names_to_use.size();
    Array<std::string,1> gas_names_this(std::move(gas_names_to_use), {n_gas});
    this->gas_names = gas_names_this;

    // Initialize the gas optics object, keeping only those gases known to the
    // gas optics and also present in the host model.
    // Add an offset to the indexing to interface the negative ranging of fortran.
    Array<Float,3> vmr_ref_red({vmr_ref.dim(1), n_gas + 1, vmr_ref.dim(3)});
    vmr_ref_red.set_offsets({0, -1, 0});

    // Gas 0 is used in single-key species method, set to 1.0 (col_dry)
    for (int i1=1; i1<=vmr_ref_red.dim(1); ++i1)
        for (int i3=1; i3<=vmr_ref_red.dim(3); ++i3)
            vmr_ref_red({i1, 0, i3}) = vmr_ref({i1, 1, i3});

    for (int i=1; i<=n_gas; ++i)
    {
        int idx = find_index(gas_names, this->gas_names({i}));
        for (int i1=1; i1<=vmr_ref_red.dim(1); ++i1)
            for (int i3=1; i3<=vmr_ref_red.dim(3); ++i3)
                vmr_ref_red({i1, i, i3}) = vmr_ref({i1, idx+1, i3}); // CvH: why +1?
    }

    this->vmr_ref = std::move(vmr_ref_red);

    // Reduce minor arrays so variables only contain minor gases that are available.
    // Reduce size of minor Arrays.
    Array<std::string, 1> minor_gases_lower_red;
    Array<std::string, 1> scaling_gas_lower_red;
    Array<std::string, 1> minor_gases_upper_red;
    Array<std::string, 1> scaling_gas_upper_red;

    reduce_minor_arrays(
            available_gases,
            gas_names,
            gas_minor, identifier_minor,
            kminor_lower,
            minor_gases_lower,
            minor_limits_gpt_lower,
            minor_scales_with_density_lower,
            scaling_gas_lower,
            scale_by_complement_lower,
            kminor_start_lower,
            this->kminor_lower,
            minor_gases_lower_red,
            this->minor_limits_gpt_lower,
            this->minor_scales_with_density_lower,
            scaling_gas_lower_red,
            this->scale_by_complement_lower,
            this->kminor_start_lower);

    reduce_minor_arrays(
            available_gases,
            gas_names,
            gas_minor,
            identifier_minor,
            kminor_upper,
            minor_gases_upper,
            minor_limits_gpt_upper,
            minor_scales_with_density_upper,
            scaling_gas_upper,
            scale_by_complement_upper,
            kminor_start_upper,
            this->kminor_upper,
            minor_gases_upper_red,
            this->minor_limits_gpt_upper,
            this->minor_scales_with_density_upper,
            scaling_gas_upper_red,
            this->scale_by_complement_upper,
            this->kminor_start_upper);

    this->first_last_minor_lower.set_dims({2,kmajor.dim(1)});
    this->first_last_minor_upper.set_dims({2,kmajor.dim(1)});
    invert_minor_gpt_limits(this->minor_limits_gpt_lower, this->first_last_minor_lower);
    invert_minor_gpt_limits(this->minor_limits_gpt_upper, this->first_last_minor_upper);

    // Arrays not reduced by the presence, or lack thereof, of a gas
    this->press_ref = press_ref;
    this->temp_ref = temp_ref;

    // Reshaping according to new dimension ordering since v1.5
    this->kmajor.set_dims({kmajor.dim(4), kmajor.dim(2), kmajor.dim(3), kmajor.dim(1)});
    for (int i4=1; i4<=this->kmajor.dim(4); ++i4)
        for (int i3=1; i3<=this->kmajor.dim(3); ++i3)
            for (int i2=1; i2<=this->kmajor.dim(2); ++i2)
                for (int i1=1; i1<=this->kmajor.dim(1); ++i1)
                    this->kmajor({i1, i2, i3, i4}) = kmajor({i4, i2, i3, i1});

    // Reshaping according to new 1.5 release.
    // Create a new vector that consists of rayl_lower and rayl_upper stored in one variable.
    if (rayl_lower.size() > 0)
    {
        this->krayl.set_dims({rayl_lower.dim(3), rayl_lower.dim(2), rayl_lower.dim(1), 2});
        for (int i3=1; i3<=this->krayl.dim(3); ++i3)
            for (int i2=1; i2<=this->krayl.dim(2); ++i2)
                for (int i1=1; i1<=this->krayl.dim(1); ++i1)
                {
                    this->krayl({i1, i2, i3, 1}) = rayl_lower({i3, i2, i1});
                    this->krayl({i1, i2, i3, 2}) = rayl_upper({i3, i2, i1});
                }
    }

    // ---- post processing ----
    //  creates log reference pressure
    this->press_ref_log = this->press_ref;
    for (int i1=1; i1<=this->press_ref_log.dim(1); ++i1)
        this->press_ref_log({i1}) = std::log(this->press_ref_log({i1}));

    // log scale of reference pressure
    this->press_ref_trop_log = std::log(press_ref_trop);

    // Get index of gas (if present) for determining col_gas
    create_idx_minor(
            this->gas_names, gas_minor, identifier_minor, minor_gases_lower_red, this->idx_minor_lower);
    create_idx_minor(
            this->gas_names, gas_minor, identifier_minor, minor_gases_upper_red, this->idx_minor_upper);

    // Get index of gas (if present) that has special treatment in density scaling
    create_idx_minor_scaling(
            this->gas_names, scaling_gas_lower_red, this->idx_minor_scaling_lower);
    create_idx_minor_scaling(
            this->gas_names, scaling_gas_upper_red, this->idx_minor_scaling_upper);

    // Create flavor list.
    // Reduce (remap) key_species list; checks that all key gases are present in incoming
    Array<int,3> key_species_red;
    Array<Bool,1> key_species_present_init;

    create_key_species_reduce(
            gas_names, this->gas_names, key_species, key_species_red, key_species_present_init);

    check_key_species_present_init(gas_names, key_species_present_init);

    // create flavor list
    create_flavor(key_species_red, this->flavor);

    // create gpoint flavor list
    create_gpoint_flavor(
            key_species_red, this->get_gpoint_bands(), this->flavor, this->gpoint_flavor);

    // minimum, maximum reference temperature, pressure -- assumes low-to-high ordering
    // for T, high-to-low ordering for p
    this->temp_ref_min = this->temp_ref({1});
    this->temp_ref_max = this->temp_ref({temp_ref.dim(1)});
    this->press_ref_min = this->press_ref({press_ref.dim(1)});
    this->press_ref_max = this->press_ref({1});

    // creates press_ref_log, temp_ref_delta
    this->press_ref_log_delta =
            (std::log(this->press_ref_min) - std::log(this->press_ref_max)) / (this->press_ref.dim(1) - 1);
    this->temp_ref_delta = (this->temp_ref_max - this->temp_ref_min) / (this->temp_ref.dim(1) - 1);

    // Which species are key in one or more bands?
    // this->flavor is an index into this->gas_names
    // if (allocated(this%is_key)) deallocate(this%is_key) ! Shouldn't ever happen...
    Array<int,1> is_key({get_ngas()}); // CvH bool, defaults to 0.?

    for (int j=1; j<=this->flavor.dim(2); ++j)
        for (int i=1; i<=this->flavor.dim(1); ++i)
        {
            if (this->flavor({i, j}) != 0)
                is_key({this->flavor({i, j})}) = true;
        }

    this->is_key = is_key;

    // copy arrays to gpu
    this->press_ref_log_gpu = this->press_ref_log;
    this->temp_ref_gpu = this->temp_ref;
    this->vmr_ref_gpu = this->vmr_ref;
    this->flavor_gpu = this->flavor;
    this->gpoint_flavor_gpu = this->gpoint_flavor;
    this->kmajor_gpu = this->kmajor;
    this->krayl_gpu = this->krayl;
    this->kminor_lower_gpu = this->kminor_lower;
    this->kminor_upper_gpu = this->kminor_upper;
    this->minor_limits_gpt_lower_gpu = this->minor_limits_gpt_lower;
    this->minor_limits_gpt_upper_gpu = this->minor_limits_gpt_upper;
    this->first_last_minor_lower_gpu = this->first_last_minor_lower;
    this->first_last_minor_upper_gpu = this->first_last_minor_upper;
    this->minor_scales_with_density_lower_gpu = this->minor_scales_with_density_lower;
    this->minor_scales_with_density_upper_gpu = this->minor_scales_with_density_upper;
    this->scale_by_complement_lower_gpu = this->scale_by_complement_lower;
    this->scale_by_complement_upper_gpu = this->scale_by_complement_upper;
    this->idx_minor_lower_gpu = this->idx_minor_lower;
    this->idx_minor_upper_gpu = this->idx_minor_upper;
    this->idx_minor_scaling_lower_gpu = this->idx_minor_scaling_lower;
    this->idx_minor_scaling_upper_gpu = this->idx_minor_scaling_upper;
    this->kminor_start_lower_gpu = this->kminor_start_lower;
    this->kminor_start_upper_gpu = this->kminor_start_upper;
    this->totplnk_gpu = this->totplnk;
    this->planck_frac_gpu = this->planck_frac;
}

    
void Gas_optics_rrtmgp_rt::find_relevant_gases_gpt(
    const int igpt,
    std::vector<int>& gases)
{
    // In any case, we want indices 0 and 1
    gases.push_back(0);
    gases.push_back(1);

    for (int iflav=1; iflav<=2; ++iflav)
    {
        const int flav = gpoint_flavor({iflav, igpt});
        for (int igas=1; igas<=2; ++igas)
        {
            const int gas = flavor({igas, flav});
            add_if_not_in_vector(gases, gas);
        }
    }

    const int minor_start_lower = first_last_minor_lower({1, igpt});
    const int minor_end_lower   = first_last_minor_lower({2, igpt});

    for (int imnr=minor_start_lower+1; imnr<=minor_end_lower+1; ++imnr)
    {
        const int minor_gas = idx_minor_lower({imnr});
        add_if_not_in_vector(gases, minor_gas);

        if ( (minor_scales_with_density_lower({imnr})) && (idx_minor_scaling_lower({imnr}) > 0))
        {
            const int scale_gas = idx_minor_scaling_lower({imnr});
            add_if_not_in_vector(gases, scale_gas);
        }
    }

    const int minor_start_upper = first_last_minor_upper({1, igpt});
    const int minor_end_upper   = first_last_minor_upper({2, igpt});

    for (int imnr=minor_start_upper+1; imnr<=minor_end_upper+1; ++imnr)
    {
        const int minor_gas = idx_minor_upper({imnr});
        add_if_not_in_vector(gases, minor_gas);

        if ( (minor_scales_with_density_upper({imnr})) && (idx_minor_scaling_upper({imnr}) > 0))
        {
            const int scale_gas = idx_minor_scaling_upper({imnr});
            add_if_not_in_vector(gases, scale_gas);
        }
    }

}

__global__
void fill_gases_kernel(
        const int col_s, const int ncol_sub, const int ncol, const int nlay, 
        const int dim1, const int dim2, const int ngas, const int igas,
        const Float* __restrict__ vmr_in,
        Float* __restrict__ col_gas, const Float* __restrict__ col_dry)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( ( icol < ncol_sub) && (ilay < nlay) )
    {
        const int idx_in = icol + col_s + ilay*ncol;
        const int idx_out = icol + ilay*ncol_sub + igas*ncol_sub*nlay;
        if (igas > 0)
        {
            Float vmr_out;
            if (dim1 == 1 && dim2 == 1)
            {
                 vmr_out = vmr_in[0];
            }
            else if (dim1 == 1)
            {
                 vmr_out = vmr_in[ilay];
            }
            else
            {
                vmr_out = vmr_in[idx_in];
            }
            col_gas[idx_out] = vmr_out * col_dry[idx_in];
        }
        else if (igas == 0)
        {
            col_gas[idx_out] = col_dry[idx_in];
        }
    }
}


__global__
void compute_delta_plev(
        const int ncol, const int nlay,
        const Float* __restrict__ plev,
        Float* __restrict__ delta_plev)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;

        delta_plev[idx] = abs(plev[idx] - plev[idx + ncol]);

        // delta_plev({icol, ilay}) = std::abs(plev({icol, ilay}) - plev({icol, ilay+1}));
    }
}


__global__
void compute_m_air(
        const int ncol, const int nlay,
        const Float* __restrict__ vmr_h2o,
        Float* __restrict__ m_air)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    constexpr Float m_dry = 0.028964;
    constexpr Float m_h2o = 0.018016;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;

        m_air[idx] = (m_dry + m_h2o * vmr_h2o[idx]) / (Float(1.) + vmr_h2o[idx]);
    }
}


__global__
void compute_col_dry(
        const int ncol, const int nlay,
        const Float* __restrict__ delta_plev, const Float* __restrict__ m_air, const Float* __restrict__ vmr_h2o,
        Float* __restrict__ col_dry)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    constexpr Float g0 = 9.80665;
    constexpr Float avogad = 6.02214076e23;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;

        col_dry[idx] = Float(10.) * delta_plev[idx] * avogad / (Float(1000.)*m_air[idx]*Float(100.)*g0);
        col_dry[idx] /= (Float(1.) + vmr_h2o[idx]);
    }
}


// Calculate the molecules of dry air.

void Gas_optics_rrtmgp_rt::get_col_dry(
        Array_gpu<Float,2>& col_dry, const Array_gpu<Float,2>& vmr_h2o,
        const Array_gpu<Float,2>& plev)
{
    Array_gpu<Float,2> delta_plev({col_dry.dim(1), col_dry.dim(2)});
    Array_gpu<Float,2> m_air     ({col_dry.dim(1), col_dry.dim(2)});

    const int block_lay = 16;
    const int block_col = 16;

    const int nlay = col_dry.dim(2);
    const int ncol = col_dry.dim(1);

    const int grid_col = ncol/block_col + (ncol%block_col > 0);
    const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

    dim3 grid_gpu(grid_col, grid_lay);
    dim3 block_gpu(block_col, block_lay);

    compute_delta_plev<<<grid_gpu, block_gpu>>>(
            ncol, nlay,
            plev.ptr(),
            delta_plev.ptr());

    compute_m_air<<<grid_gpu, block_gpu>>>(
            ncol, nlay,
            vmr_h2o.ptr(),
            m_air.ptr());

    compute_col_dry<<<grid_gpu, block_gpu>>>(
            ncol, nlay,
            delta_plev.ptr(), m_air.ptr(), vmr_h2o.ptr(),
            col_dry.ptr());
}


// Gas optics solver longwave variant.

void Gas_optics_rrtmgp_rt::gas_optics(
        const int igpt,
        const int col_s, const int ncol_sub, const int ncol,
        const Array_gpu<Float,2>& play,
        const Array_gpu<Float,2>& plev,
        const Array_gpu<Float,2>& tlay,
        const Array_gpu<Float,1>& tsfc,
        const Gas_concs_gpu& gas_desc,
        std::unique_ptr<Optical_props_arry_rt>& optical_props,
        Source_func_lw_rt& sources,
        const Array_gpu<Float,2>& col_dry,
        const Array_gpu<Float,2>& tlev)
{
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();

    if (jtemp.size()==0)
    {
        this->jtemp.set_dims({play.dim(1), play.dim(2)});
        this->jpress.set_dims({play.dim(1), play.dim(2)});
        this->tropo.set_dims({play.dim(1), play.dim(2)});
        this->fmajor.set_dims({2, 2, 2, play.dim(1), play.dim(2)});
        this->jeta.set_dims({2, play.dim(1), play.dim(2)});
    }

    // Gas optics.
    compute_gas_taus(
            1, ncol_sub, ncol, nlay, ngpt, nband, igpt,
            play, plev, tlay, gas_desc,
            optical_props,
            this->jtemp, this->jpress, this->jeta, this->tropo, this->fmajor,
            col_dry);

    // External sources.
    source(
            ncol, nlay, nband, ngpt, igpt,
            play, plev, tlay, tsfc,
            this->jtemp, this->jpress, this->jeta, this->tropo, this->fmajor,
            sources, tlev);
}


// Gas optics solver shortwave variant.

void Gas_optics_rrtmgp_rt::gas_optics(
        const int igpt, 
        const int col_s, const int ncol_sub, const int ncol,
        const Array_gpu<Float,2>& play,
        const Array_gpu<Float,2>& plev,
        const Array_gpu<Float,2>& tlay,
        const Gas_concs_gpu& gas_desc,
        std::unique_ptr<Optical_props_arry_rt>& optical_props,
        Array_gpu<Float,1>& toa_src,
        const Array_gpu<Float,2>& col_dry)
{
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();

    if (this->jtemp.size()==0)
    {
        this->jtemp.set_dims({ncol_sub, nlay});
        this->jpress.set_dims({ncol_sub, nlay});
        this->tropo.set_dims({ncol_sub, nlay});
        this->fmajor.set_dims({2, 2, 2, ncol_sub, nlay});
        this->jeta.set_dims({2, ncol_sub, nlay});
    }

    // Gas optics.
    compute_gas_taus(
            col_s, ncol_sub, ncol, nlay, ngpt, nband, igpt,
            play, plev, tlay, gas_desc,
            optical_props,
            this->jtemp, this->jpress, this->jeta, this->tropo, this->fmajor,
            col_dry);

    // External source function is constant.
    spread_col(ncol_sub, igpt, toa_src, this->solar_source_gpu);
}



void Gas_optics_rrtmgp_rt::compute_gas_taus(
        const int col_s, const int ncol_sub, const int ncol, const int nlay,
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
        const Array_gpu<Float,2>& col_dry)
{
    // CvH add all the checking...
    const int ngas = this->get_ngas();
    const int nflav = this->get_nflav();
    const int neta = this->get_neta();
    const int npres = this->get_npres();
    const int ntemp = this->get_ntemp();

    const int nminorlower = this->minor_scales_with_density_lower.dim(1);
    const int nminorklower = this->kminor_lower.dim(3);
    const int nminorupper = this->minor_scales_with_density_upper.dim(1);
    const int nminorkupper = this->kminor_upper.dim(3);

    if (fminor.size()==0)
    {
        this->col_gas.set_dims({ncol_sub, nlay, this->get_ngas()+1});
        this->col_gas.set_offsets({0, 0, -1});
        this->col_mix.set_dims({2, ncol_sub, nlay});
        this->fminor.set_dims({2, 2, ncol_sub, nlay});
    }

    const int block_lay = 16;
    const int block_col = 16;

    const int grid_col = ncol_sub/block_col + (ncol_sub%block_col > 0);
    const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

    dim3 grid_gpu(grid_col, grid_lay);
    dim3 block_gpu(block_col, block_lay);

    // what gases do we need?
    std::vector<int> gases;
    find_relevant_gases_gpt(igpt, gases);

    for (int i=0; i<gases.size(); ++i)
    {
        const int igas = gases[i];
        const Array_gpu<Float,2>& vmr_2d = igas > 0 ? gas_desc.get_vmr(this->gas_names({igas})) : gas_desc.get_vmr(this->gas_names({1}));
        fill_gases_kernel<<<grid_gpu, block_gpu>>>(
            col_s, ncol_sub, ncol, nlay, vmr_2d.dim(1), vmr_2d.dim(2), ngas, igas, vmr_2d.ptr(), col_gas.ptr(), col_dry.ptr());
    }
    
    Gas_optics_rrtmgp_kernels_cuda_rt::interpolation(
            col_s, ncol_sub, ncol, nlay, igpt,
            ngas, nflav, neta, npres, ntemp,
            gpoint_flavor_gpu.ptr(),
            flavor_gpu.ptr(),
            press_ref_log_gpu.ptr(),
            temp_ref_gpu.ptr(),
            this->press_ref_log_delta,
            this->temp_ref_min,
            this->temp_ref_delta,
            this->press_ref_trop_log,
            vmr_ref_gpu.ptr(),
            play.ptr(),
            tlay.ptr(),
            col_gas.ptr(),
            jtemp.ptr(),
            fmajor.ptr(), fminor.ptr(),
            col_mix.ptr(),
            tropo.ptr(),
            jeta.ptr(), jpress.ptr());

    this->idx_h2o=-1;
    for (int i=1; i<=this->gas_names.dim(1); ++i)
        if (gas_names({i}) == "h2o")
        {
            this->idx_h2o = i;
            break;
        }

    if (this->idx_h2o == -1)
        throw std::runtime_error("idx_h2o cannot be found");

    bool has_rayleigh = (this->krayl.size() > 0);

    if (has_rayleigh)
    {
        Array_gpu<Float,2> tau({ncol_sub, nlay});
        Array_gpu<Float,2> tau_rayleigh({ncol_sub, nlay});
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(nlay, ncol_sub, tau.ptr());
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(ncol_sub, nlay, tau_rayleigh.ptr());

        Gas_optics_rrtmgp_kernels_cuda_rt::compute_tau_absorption(
                col_s, ncol_sub, ncol, nlay, nband, ngpt, igpt,
                ngas, nflav, neta, npres, ntemp,
                nminorlower, nminorklower,
                nminorupper, nminorkupper,
                idx_h2o,
                this->get_band_lims_gpoint_gpu().ptr(),
                kmajor_gpu.ptr(),
                kminor_lower_gpu.ptr(),
                kminor_upper_gpu.ptr(),
                minor_limits_gpt_lower_gpu.ptr(),
                minor_limits_gpt_upper_gpu.ptr(),
                first_last_minor_lower_gpu.ptr(),
                first_last_minor_upper_gpu.ptr(),
                minor_scales_with_density_lower_gpu.ptr(),
                minor_scales_with_density_upper_gpu.ptr(),
                scale_by_complement_lower_gpu.ptr(),
                scale_by_complement_upper_gpu.ptr(),
                idx_minor_lower_gpu.ptr(),
                idx_minor_upper_gpu.ptr(),
                idx_minor_scaling_lower_gpu.ptr(),
                idx_minor_scaling_upper_gpu.ptr(),
                kminor_start_lower_gpu.ptr(),
                kminor_start_upper_gpu.ptr(),
                tropo.ptr(),
                col_mix.ptr(), fmajor.ptr(), fminor.ptr(),
                play.ptr(), tlay.ptr(), col_gas.ptr(),
                jeta.ptr(), jtemp.ptr(), jpress.ptr(),
                tau.ptr());

        Gas_optics_rrtmgp_kernels_cuda_rt::compute_tau_rayleigh(
                col_s, ncol_sub, ncol, nlay, nband, ngpt, igpt,
                ngas, nflav, neta, npres, ntemp,
                this->get_gpoint_bands_gpu().ptr(),
                this->get_band_lims_gpoint_gpu().ptr(),
                krayl_gpu.ptr(),
                idx_h2o, col_dry.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(), tropo.ptr(), jtemp.ptr(),
                tau_rayleigh.ptr());

        Gas_optics_rrtmgp_kernels_cuda_rt::combine_abs_and_rayleigh(
                col_s, ncol_sub, ncol, nlay,
                tau.ptr(), tau_rayleigh.ptr(),
                optical_props->get_tau().ptr(), optical_props->get_ssa().ptr(), optical_props->get_g().ptr());
    }
    else
    {
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(ncol, nlay, optical_props->get_tau().ptr());

        Gas_optics_rrtmgp_kernels_cuda_rt::compute_tau_absorption(
                col_s, ncol_sub, ncol, nlay, nband, ngpt, igpt,
                ngas, nflav, neta, npres, ntemp,
                nminorlower, nminorklower,
                nminorupper, nminorkupper,
                idx_h2o,
                this->get_band_lims_gpoint_gpu().ptr(),
                kmajor_gpu.ptr(),
                kminor_lower_gpu.ptr(),
                kminor_upper_gpu.ptr(),
                minor_limits_gpt_lower_gpu.ptr(),
                minor_limits_gpt_upper_gpu.ptr(),
                first_last_minor_lower_gpu.ptr(),
                first_last_minor_upper_gpu.ptr(),
                minor_scales_with_density_lower_gpu.ptr(),
                minor_scales_with_density_upper_gpu.ptr(),
                scale_by_complement_lower_gpu.ptr(),
                scale_by_complement_upper_gpu.ptr(),
                idx_minor_lower_gpu.ptr(),
                idx_minor_upper_gpu.ptr(),
                idx_minor_scaling_lower_gpu.ptr(),
                idx_minor_scaling_upper_gpu.ptr(),
                kminor_start_lower_gpu.ptr(),
                kminor_start_upper_gpu.ptr(),
                tropo.ptr(),
                col_mix.ptr(), fmajor.ptr(), fminor.ptr(),
                play.ptr(), tlay.ptr(), col_gas.ptr(),
                jeta.ptr(), jtemp.ptr(), jpress.ptr(),
                optical_props->get_tau().ptr());
    }
}

// void Gas_optics_rrtmgp_rt::combine_abs_and_rayleigh(
//         const Array_gpu<Float,2>& tau,
//         const Array_gpu<Float,2>& tau_rayleigh,
//         std::unique_ptr<Optical_props_arry_rt>& optical_props)
// {
//     int ncol = tau.dim(1);
//     int nlay = tau.dim(2);
//
//     Gas_optics_rrtmgp_kernels_cuda_rt::combine_abs_and_rayleigh(
//             ncol, nlay,
//             tau, tau_rayleigh,
//             optical_props->get_tau(), optical_props->get_ssa(), optical_props->get_g());
// }

void Gas_optics_rrtmgp_rt::source(
        const int ncol, const int nlay, const int nbnd, const int ngpt, const int igpt,
        const Array_gpu<Float,2>& play, const Array_gpu<Float,2>& plev,
        const Array_gpu<Float,2>& tlay, const Array_gpu<Float,1>& tsfc,
        const Array_gpu<int,2>& jtemp, const Array_gpu<int,2>& jpress,
        const Array_gpu<int,3>& jeta, const Array_gpu<Bool,2>& tropo,
        const Array_gpu<Float,5>& fmajor,
        Source_func_lw_rt& sources,
        const Array_gpu<Float,2>& tlev)
{
    const int nflav = this->get_nflav();
    const int neta = this->get_neta();
    const int npres = this->get_npres();
    const int ntemp = this->get_ntemp();
    const int nPlanckTemp = this->get_nPlanckTemp();
    auto gpoint_bands = this->get_gpoint_bands_gpu();
    auto band_lims_gpoint = this->get_band_lims_gpoint_gpu();

    int sfc_lay = play({1, 1}) > play({1, nlay}) ? 1 : nlay;

    Gas_optics_rrtmgp_kernels_cuda_rt::Planck_source(
            ncol, nlay, nbnd, ngpt, igpt,
            nflav, neta, npres, ntemp, nPlanckTemp,
            tlay.ptr(), tlev.ptr(), tsfc.ptr(), sfc_lay,
            fmajor.ptr(), jeta.ptr(), tropo.ptr(), jtemp.ptr(), jpress.ptr(),
            gpoint_bands.ptr(), band_lims_gpoint.ptr(), this->planck_frac_gpu.ptr(), this->temp_ref_min,
            this->totplnk_delta, this->totplnk_gpu.ptr(),
            sources.get_sfc_source().ptr(), sources.get_lay_source().ptr(), sources.get_lev_source_inc().ptr(),
            sources.get_lev_source_dec().ptr(), sources.get_sfc_source_jac().ptr());
}



void Gas_optics_rrtmgp_rt::set_solar_variability(
        const Float mg_index, const Float sb_index)
{
    constexpr Float a_offset = Float(0.1495954);
    constexpr Float b_offset = Float(0.00066696);

    for (int igpt=1; igpt<=this->solar_source_quiet.dim(1); ++igpt)
    {
        this->solar_source({igpt}) = this->solar_source_quiet({igpt})
                + (mg_index - a_offset) * this->solar_source_facular({igpt})
                + (sb_index - b_offset) * this->solar_source_sunspot({igpt});
    }
    this->solar_source_gpu = this->solar_source;
}



Float Gas_optics_rrtmgp_rt::get_tsi() const
{
    const int n_gpt = this->get_ngpt();

    Float tsi = 0.;
    for (int igpt=1; igpt<=n_gpt; ++igpt)
        tsi += this->solar_source({igpt});

    return tsi;
}

Float Gas_optics_rrtmgp_rt::band_source(const int gpt_start, const int gpt_end) const
{
    Float bnd_src = 0.;
    for (int igpt=gpt_start; igpt<=gpt_end; ++igpt)
        bnd_src += this->solar_source({igpt});

    return bnd_src;
}


