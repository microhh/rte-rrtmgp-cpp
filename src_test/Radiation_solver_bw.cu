/*
 * This file is imported from MicroHH (https://github.com/earth-system-radiation/earth-system-radiation)
 * and is adapted for the testing of the C++ interface to the
 * RTE+RRTMGP radiation code.
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/algorithm/string.hpp>
#include <cmath>
#include <numeric>
#include <curand_kernel.h>

#include "Radiation_solver_bw.h"
#include "Status.h"
#include "Netcdf_interface.h"

#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics_rrtmgp_rt.h"
#include "Optical_props_rt.h"
#include "Source_functions_rt.h"
#include "Fluxes_rt.h"
#include "Rte_lw_rt.h"
#include "Rte_sw_rt.h"

#include "subset_kernels_cuda.h"
#include "gas_optics_rrtmgp_kernels_cuda_rt.h"
#include "gpt_combine_kernels_cuda_rt.h"


namespace
{
    // spectral albedo functions estimated from http://gsp.humboldt.edu/olm/Courses/GSP_216/lessons/reflectance.html
    __device__
    Float get_grass_alb_proc(const Float wv)
    {
        if (wv < 500)
        {
            return Float(3.0) + Float(0.0065) * (wv - Float(300.));
        }
        else if (wv < 550)
        {
            return Float(4.3) + Float(0.216) * (wv - Float(500.));
        }
        else if (wv < 580)
        {
            return Float(15.1) - Float(0.13) * (wv - Float(550.));
        }
        else if (wv < 680)
        {
            return Float(12.) - Float(0.083) * (wv - Float(580.));
        }
        else if (wv < 750)
        {
            return Float(4.5) - Float(0.5) * (wv - Float(680.));
        }
        else
        {
            return Float(45);
        }
    }

    __device__
    Float get_soil_alb_proc(const Float wv)
    {
        if (wv < 400)
        {
            return Float(0.4);
        }
        else
        {
            return Float(0.4) + Float(0.085) * (wv - Float(400.));
        }
    }

    __device__
    Float get_concrete_alb_proc(const Float wv)
    {
        if (wv < 600)
        {
            return Float(9) + Float(0.0666666) * (wv - Float(300));
        }
        else
        {
            return Float(30);
        }
    }

    __device__
    Float mean_albedo(const Float wv1, const Float wv2,  Float (*f_albedo)(Float))
    {
        const int nwv = 100;
        const Float dwv = (wv2 - wv1)/Float(nwv);
        Float albedo = Float(0.);
        for (int i=0; i<nwv; ++i)
            albedo += f_albedo(wv1 + i*dwv);
        return albedo / Float(nwv) * Float(0.01);
    }

    __global__
    void spectral_albedo_kernel(const int ncol, const Float wv1, const Float wv2,
                         const Float* __restrict__ land_use_map,
                         Float* __restrict__ albedo)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x;
        if ( (i<ncol) )
        {
            if (land_use_map[i] == 0)
            {
                albedo[i] = Float(0.25);
            }
            else if (land_use_map[i] >= 1 && land_use_map[i] <= 2)
            {
                albedo[i] = mean_albedo(wv1, wv2, &get_grass_alb_proc) * (land_use_map[i]-1) + mean_albedo(wv1, wv2, &get_soil_alb_proc) * (Float(2.)-land_use_map[i]);
            }
            else if (land_use_map[i] == 3)
            {
                albedo[i] = mean_albedo(wv1, wv2, &get_concrete_alb_proc);
            }
       }
    }

    void spectral_albedo(const int ncol, const Float wv1, const Float wv2,
                         const Array_gpu<Float,1>& land_use_map,
                         Array_gpu<Float,2>& albedo)
    {
        const int block_col = 16;
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, 1);
        dim3 block_gpu(block_col, 1);
        spectral_albedo_kernel<<<grid_gpu, block_gpu>>>(
            ncol, wv1, wv2, land_use_map.ptr(), albedo.ptr());
    }

    std::vector<std::string> get_variable_string(
            const std::string& var_name,
            std::vector<int> i_count,
            Netcdf_handle& input_nc,
            const int string_len,
            bool trim=true)
    {
        // Multiply all elements in i_count.
        int total_count = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Add the string length as the rightmost dimension.
        i_count.push_back(string_len);

        // Read the entire char array;
        std::vector<char> var_char;
        var_char = input_nc.get_variable<char>(var_name, i_count);

        std::vector<std::string> var;

        for (int n=0; n<total_count; ++n)
        {
            std::string s(var_char.begin()+n*string_len, var_char.begin()+(n+1)*string_len);
            if (trim)
                boost::trim(s);
            var.push_back(s);
        }

        return var;
    }


    Gas_optics_rrtmgp_rt load_and_init_gas_optics(
            const Gas_concs_gpu& gas_concs,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(coef_file, Netcdf_mode::Read);

        // Read k-distribution information.
        const int n_temps = coef_nc.get_dimension_size("temperature");
        const int n_press = coef_nc.get_dimension_size("pressure");
        const int n_absorbers = coef_nc.get_dimension_size("absorber");
        const int n_char = coef_nc.get_dimension_size("string_len");
        const int n_minorabsorbers = coef_nc.get_dimension_size("minor_absorber");
        const int n_extabsorbers = coef_nc.get_dimension_size("absorber_ext");
        const int n_mixingfracs = coef_nc.get_dimension_size("mixing_fraction");
        const int n_layers = coef_nc.get_dimension_size("atmos_layer");
        const int n_bnds = coef_nc.get_dimension_size("bnd");
        const int n_gpts = coef_nc.get_dimension_size("gpt");
        const int n_pairs = coef_nc.get_dimension_size("pair");
        const int n_minor_absorber_intervals_lower = coef_nc.get_dimension_size("minor_absorber_intervals_lower");
        const int n_minor_absorber_intervals_upper = coef_nc.get_dimension_size("minor_absorber_intervals_upper");
        const int n_contributors_lower = coef_nc.get_dimension_size("contributors_lower");
        const int n_contributors_upper = coef_nc.get_dimension_size("contributors_upper");

        // Read gas names.
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});

        Array<int,3> key_species(
                coef_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
                {2, n_layers, n_bnds});
        Array<Float,2> band_lims(coef_nc.get_variable<Float>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<Float,1> press_ref(coef_nc.get_variable<Float>("press_ref", {n_press}), {n_press});
        Array<Float,1> temp_ref(coef_nc.get_variable<Float>("temp_ref", {n_temps}), {n_temps});

        Float temp_ref_p = coef_nc.get_variable<Float>("absorption_coefficient_ref_P");
        Float temp_ref_t = coef_nc.get_variable<Float>("absorption_coefficient_ref_T");
        Float press_ref_trop = coef_nc.get_variable<Float>("press_ref_trop");

        Array<Float,3> kminor_lower(
                coef_nc.get_variable<Float>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<Float,3> kminor_upper(
                coef_nc.get_variable<Float>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
                {n_contributors_upper, n_mixingfracs, n_temps});

        Array<std::string,1> gas_minor(get_variable_string("gas_minor", {n_minorabsorbers}, coef_nc, n_char),
                                       {n_minorabsorbers});

        Array<std::string,1> identifier_minor(
                get_variable_string("identifier_minor", {n_minorabsorbers}, coef_nc, n_char), {n_minorabsorbers});

        Array<std::string,1> minor_gases_lower(
                get_variable_string("minor_gases_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> minor_gases_upper(
                get_variable_string("minor_gases_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,2> minor_limits_gpt_lower(
                coef_nc.get_variable<int>("minor_limits_gpt_lower", {n_minor_absorber_intervals_lower, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_lower});
        Array<int,2> minor_limits_gpt_upper(
                coef_nc.get_variable<int>("minor_limits_gpt_upper", {n_minor_absorber_intervals_upper, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_upper});

        Array<Bool,1> minor_scales_with_density_lower(
                coef_nc.get_variable<Bool>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<Bool,1> minor_scales_with_density_upper(
                coef_nc.get_variable<Bool>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<Bool,1> scale_by_complement_lower(
                coef_nc.get_variable<Bool>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<Bool,1> scale_by_complement_upper(
                coef_nc.get_variable<Bool>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<std::string,1> scaling_gas_lower(
                get_variable_string("scaling_gas_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> scaling_gas_upper(
                get_variable_string("scaling_gas_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,1> kminor_start_lower(
                coef_nc.get_variable<int>("kminor_start_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> kminor_start_upper(
                coef_nc.get_variable<int>("kminor_start_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<Float,3> vmr_ref(
                coef_nc.get_variable<Float>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                {n_layers, n_extabsorbers, n_temps});

        Array<Float,4> kmajor(
                coef_nc.get_variable<Float>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press+1, n_temps});

        // Keep the size at zero, if it does not exist.
        Array<Float,3> rayl_lower;
        Array<Float,3> rayl_upper;

        if (coef_nc.variable_exists("rayl_lower"))
        {
            rayl_lower.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_upper.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_lower = coef_nc.get_variable<Float>("rayl_lower", {n_temps, n_mixingfracs, n_gpts});
            rayl_upper = coef_nc.get_variable<Float>("rayl_upper", {n_temps, n_mixingfracs, n_gpts});
        }

        // Is it really LW if so read these variables as well.
        if (coef_nc.variable_exists("totplnk"))
        {
            int n_internal_sourcetemps = coef_nc.get_dimension_size("temperature_Planck");

            Array<Float,2> totplnk(
                    coef_nc.get_variable<Float>( "totplnk", {n_bnds, n_internal_sourcetemps}),
                    {n_internal_sourcetemps, n_bnds});
            Array<Float,4> planck_frac(
                    coef_nc.get_variable<Float>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                    {n_gpts, n_mixingfracs, n_press+1, n_temps});

            // Construct the k-distribution.
            return Gas_optics_rrtmgp_rt(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    totplnk,
                    planck_frac,
                    rayl_lower,
                    rayl_upper);
        }
        else
        {
            Array<Float,1> solar_src_quiet(
                    coef_nc.get_variable<Float>("solar_source_quiet", {n_gpts}), {n_gpts});
            Array<Float,1> solar_src_facular(
                    coef_nc.get_variable<Float>("solar_source_facular", {n_gpts}), {n_gpts});
            Array<Float,1> solar_src_sunspot(
                    coef_nc.get_variable<Float>("solar_source_sunspot", {n_gpts}), {n_gpts});

            Float tsi = coef_nc.get_variable<Float>("tsi_default");
            Float mg_index = coef_nc.get_variable<Float>("mg_default");
            Float sb_index = coef_nc.get_variable<Float>("sb_default");

            return Gas_optics_rrtmgp_rt(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    solar_src_quiet,
                    solar_src_facular,
                    solar_src_sunspot,
                    tsi,
                    mg_index,
                    sb_index,
                    rayl_lower,
                    rayl_upper);
        }
        // End reading of k-distribution.
    }


    Cloud_optics_rt load_and_init_cloud_optics(
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(coef_file, Netcdf_mode::Read);

        // Read look-up table coefficient dimensions
        int n_band     = coef_nc.get_dimension_size("nband");
        int n_rghice   = coef_nc.get_dimension_size("nrghice");
        int n_size_liq = coef_nc.get_dimension_size("nsize_liq");
        int n_size_ice = coef_nc.get_dimension_size("nsize_ice");

        Array<Float,2> band_lims_wvn(coef_nc.get_variable<Float>("bnd_limits_wavenumber", {n_band, 2}), {2, n_band});

        // Read look-up table constants.
        Float radliq_lwr = coef_nc.get_variable<Float>("radliq_lwr");
        Float radliq_upr = coef_nc.get_variable<Float>("radliq_upr");
        Float radliq_fac = coef_nc.get_variable<Float>("radliq_fac");

        Float radice_lwr = coef_nc.get_variable<Float>("radice_lwr");
        Float radice_upr = coef_nc.get_variable<Float>("radice_upr");
        Float radice_fac = coef_nc.get_variable<Float>("radice_fac");

        Array<Float,2> lut_extliq(
                coef_nc.get_variable<Float>("lut_extliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<Float,2> lut_ssaliq(
                coef_nc.get_variable<Float>("lut_ssaliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<Float,2> lut_asyliq(
                coef_nc.get_variable<Float>("lut_asyliq", {n_band, n_size_liq}), {n_size_liq, n_band});

        Array<Float,3> lut_extice(
                coef_nc.get_variable<Float>("lut_extice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<Float,3> lut_ssaice(
                coef_nc.get_variable<Float>("lut_ssaice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<Float,3> lut_asyice(
                coef_nc.get_variable<Float>("lut_asyice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});

        return Cloud_optics_rt(
                band_lims_wvn,
                radliq_lwr, radliq_upr, radliq_fac,
                radice_lwr, radice_upr, radice_fac,
                lut_extliq, lut_ssaliq, lut_asyliq,
                lut_extice, lut_ssaice, lut_asyice);
    }

    Aerosol_optics_rt load_and_init_aerosol_optics(
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(coef_file, Netcdf_mode::Read);

        // Read look-up table coefficient dimensions
        int n_band     = coef_nc.get_dimension_size("band_sw");
        int n_hum      = coef_nc.get_dimension_size("relative_humidity");
        int n_philic = coef_nc.get_dimension_size("hydrophilic");
        int n_phobic = coef_nc.get_dimension_size("hydrophobic");

        Array<Float,2> band_lims_wvn({2, n_band});

        Array<Float,2> mext_phobic(
                coef_nc.get_variable<Float>("mass_ext_sw_hydrophobic", {n_phobic, n_band}), {n_band, n_phobic});
        Array<Float,2> ssa_phobic(
                coef_nc.get_variable<Float>("ssa_sw_hydrophobic", {n_phobic, n_band}), {n_band, n_phobic});
        Array<Float,2> g_phobic(
                coef_nc.get_variable<Float>("asymmetry_sw_hydrophobic", {n_phobic, n_band}), {n_band, n_phobic});

        Array<Float,3> mext_philic(
                coef_nc.get_variable<Float>("mass_ext_sw_hydrophilic", {n_philic, n_hum, n_band}), {n_band, n_hum, n_philic});
        Array<Float,3> ssa_philic(
                coef_nc.get_variable<Float>("ssa_sw_hydrophilic", {n_philic, n_hum, n_band}), {n_band, n_hum, n_philic});
        Array<Float,3> g_philic(
                coef_nc.get_variable<Float>("asymmetry_sw_hydrophilic", {n_philic, n_hum, n_band}), {n_band, n_hum, n_philic});

        Array<Float,1> rh_upper(
                coef_nc.get_variable<Float>("relative_humidity2", {n_hum}), {n_hum});

        return Aerosol_optics_rt(
                band_lims_wvn, rh_upper,
                mext_phobic, ssa_phobic, g_phobic,
                mext_philic, ssa_philic, g_philic);
    }
}


Radiation_solver_longwave::Radiation_solver_longwave(
        const Gas_concs_gpu& gas_concs,
        const std::string& file_name_gas,
        const std::string& file_name_cloud)
{
    // Construct the gas optics classes for the solver.
    this->kdist_gpu = std::make_unique<Gas_optics_rrtmgp_rt>(
            load_and_init_gas_optics(gas_concs, file_name_gas));

    this->cloud_optics_gpu = std::make_unique<Cloud_optics_rt>(
            load_and_init_cloud_optics(file_name_cloud));
}


void Radiation_solver_longwave::solve_gpu(
        const bool switch_fluxes,
        const bool switch_cloud_optics,
        const bool switch_output_optical,
        const bool switch_output_bnd_fluxes,
        const Gas_concs_gpu& gas_concs,
        const Array_gpu<Float,2>& p_lay, const Array_gpu<Float,2>& p_lev,
        const Array_gpu<Float,2>& t_lay, const Array_gpu<Float,2>& t_lev,
        Array_gpu<Float,2>& col_dry,
        const Array_gpu<Float,1>& t_sfc, const Array_gpu<Float,2>& emis_sfc,
        const Array_gpu<Float,2>& lwp, const Array_gpu<Float,2>& iwp,
        const Array_gpu<Float,2>& rel, const Array_gpu<Float,2>& rei,
        Array_gpu<Float,3>& tau, Array_gpu<Float,3>& lay_source,
        Array_gpu<Float,3>& lev_source_inc, Array_gpu<Float,3>& lev_source_dec, Array_gpu<Float,2>& sfc_source,
        Array_gpu<Float,2>& lw_flux_up, Array_gpu<Float,2>& lw_flux_dn, Array_gpu<Float,2>& lw_flux_net,
        Array_gpu<Float,3>& lw_bnd_flux_up, Array_gpu<Float,3>& lw_bnd_flux_dn, Array_gpu<Float,3>& lw_bnd_flux_net)
{
    const int n_col = p_lay.dim(1);
    const int n_lay = p_lay.dim(2);
    const int n_lev = p_lev.dim(2);
    const int n_gpt = this->kdist_gpu->get_ngpt();
    const int n_bnd = this->kdist_gpu->get_nband();

    const Bool top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

    optical_props = std::make_unique<Optical_props_1scl_rt>(n_col, n_lay, *kdist_gpu);
    sources = std::make_unique<Source_func_lw_rt>(n_col, n_lay, *kdist_gpu);

    if (switch_cloud_optics)
        cloud_optical_props = std::make_unique<Optical_props_1scl_rt>(n_col, n_lay, *cloud_optics_gpu);

    if (col_dry.size() == 0)
    {
        col_dry.set_dims({n_col, n_lay});
        Gas_optics_rrtmgp_rt::get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);
    }

    if (switch_fluxes)
    {
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_lev, n_col, lw_flux_up.ptr());
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_lev, n_col, lw_flux_dn.ptr());
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_lev, n_col, lw_flux_net.ptr());
    }

    const Array<int, 2>& band_limits_gpt(this->kdist_gpu->get_band_lims_gpoint());
    for (int igpt=1; igpt<=n_gpt; ++igpt)
    {
        int band = 0;
        for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
        {
            if (igpt <= band_limits_gpt({2, ibnd}))
            {
                band = ibnd;
                break;
            }
        }

        //kdist_gpu->gas_optics(
        //        igpt-1,
        //        p_lay,
        //        p_lev,
        //        t_lay,
        //        t_sfc,
        //        gas_concs,
        //        optical_props,
        //        *sources,
        //        col_dry,
        //        t_lev);

        if (switch_cloud_optics)
        {
            cloud_optics_gpu->cloud_optics(
                    band,
                    lwp,
                    iwp,
                    rel,
                    rei,
                    *cloud_optical_props);
            // cloud->delta_scale();

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_1scl_rt&>(*optical_props),
                    dynamic_cast<Optical_props_1scl_rt&>(*cloud_optical_props));
        }

        // Store the optical properties, if desired.
        if (switch_output_optical)
        {
            Gpt_combine_kernels_cuda_rt::get_from_gpoint(
                    n_col, n_lay, igpt-1, tau.ptr(), lay_source.ptr(), lev_source_inc.ptr(), lev_source_dec.ptr(),
                    optical_props->get_tau().ptr(), (*sources).get_lay_source().ptr(),
                    (*sources).get_lev_source_inc().ptr(), (*sources).get_lev_source_dec().ptr());

            Gpt_combine_kernels_cuda_rt::get_from_gpoint(
                    n_col, igpt-1, sfc_source.ptr(), (*sources).get_sfc_source().ptr());
        }


        if (switch_fluxes)
        {
            constexpr int n_ang = 1;

            std::unique_ptr<Fluxes_broadband_rt> fluxes =
                    std::make_unique<Fluxes_broadband_rt>(n_col, 1, n_lev);

            rte_lw.rte_lw(
                    optical_props,
                    top_at_1,
                    *sources,
                    emis_sfc.subset({{ {band, band}, {1, n_col}}}),
                    Array_gpu<Float,1>(), // Add an empty array, no inc_flux.
                    (*fluxes).get_flux_up(),
                    (*fluxes).get_flux_dn(),
                    n_ang);

            (*fluxes).net_flux();

            // Copy the data to the output.
            Gpt_combine_kernels_cuda_rt::add_from_gpoint(
                    n_col, n_lev, lw_flux_up.ptr(), lw_flux_dn.ptr(), lw_flux_net.ptr(),
                    (*fluxes).get_flux_up().ptr(), (*fluxes).get_flux_dn().ptr(), (*fluxes).get_flux_net().ptr());


            if (switch_output_bnd_fluxes)
            {
                Gpt_combine_kernels_cuda_rt::get_from_gpoint(
                        n_col, n_lev, igpt-1, lw_bnd_flux_up.ptr(), lw_bnd_flux_dn.ptr(), lw_bnd_flux_net.ptr(),
                        (*fluxes).get_flux_up().ptr(), (*fluxes).get_flux_dn().ptr(), (*fluxes).get_flux_net().ptr());

            }
        }
    }
}



Float get_x(const Float wv)
{
    const Float a = (wv - Float(442.0)) * ((wv < Float(442.0)) ? Float(0.0624) : Float(0.0374));
    const Float b = (wv - Float(599.8)) * ((wv < Float(599.8)) ? Float(0.0264) : Float(0.0323));
    const Float c = (wv - Float(501.1)) * ((wv < Float(501.1)) ? Float(0.0490) : Float(0.0382));
    return Float(0.362) * std::exp(Float(-0.5)*a*a) + Float(1.056) * std::exp(Float(-0.5)*b*b) - Float(0.065) * std::exp(Float(-0.5)*c*c);
}


Float get_y(const Float wv)
{
    const Float a = (wv - Float(568.8)) * ((wv < Float(568.8)) ? Float(0.0213) : Float(0.0247));
    const Float b = (wv - Float(530.9)) * ((wv < Float(530.9)) ? Float(0.0613) : Float(0.0322));
    return Float(0.821) * std::exp(Float(-0.5)*a*a) + Float(.286) * std::exp(Float(-0.5)*b*b);
}

Float get_z(const Float wv)
{
    const Float a = (wv - Float(437.0)) * ((wv < Float(437.0)) ? Float(0.0845) : Float(0.0278));
    const Float b = (wv - Float(459.0)) * ((wv < Float(459.0)) ? Float(0.0385) : Float(0.0725));
    return Float(1.217) * std::exp(Float(-0.5)*a*a) + Float(0.681) * std::exp(Float(-0.5)*b*b);
}

Float Planck(Float wv)
{
    const Float h = Float(6.62607015e-34);
    const Float c = Float(299792458.);
    const Float k = Float(1.380649e-23);
    const Float nom = 2*h*c*c / (wv*wv*wv*wv*wv);
    const Float denom = exp(h*c/(wv*k*Float(5778)))-Float(1.);
    return (nom/denom);
}


Float Planck_integrator(
        const Float wv1, const Float wv2)
{
    const int n = 100;
    const Float dwv = (wv2-wv1)/Float(n);
    Float sum = 0;
    for (int i=0; i<n; ++i)
    {
        const Float wv = (wv1 + i*dwv)*1e-9;
        sum += Planck(wv) * dwv;
    }
    return sum * Float(1e-9);
}


// Following bodhaine 1999: https://doi.org/10.1175/1520-0426(1999)016%3C1854:ORODC%3E2.0.CO;2
Float rayleigh_mean(
    const Float wv1, const Float wv2)
{
    const Float Ns = 2.546899e19;
    const Float dwv = (wv2-wv1)/100.;
    Float sigma_mean = 0;
    for (int i=0; i<100; ++i)
    {
        const Float wv = (wv1 + i*dwv);
        const Float n = 1+1e-8*(8060.77 + 2481070/(132.274-pow((wv/1e3),-2)) + 17456.3/(39.32957-pow((wv/1e3),-2)));
        const Float nom = 24*M_PI*M_PI*M_PI*pow((n*n-1),2);
        const Float denom = pow((wv/1e7),4) * Ns*Ns * pow((n*n +2), 2);
        sigma_mean += nom/denom * 1.055;
    }
    return sigma_mean / 100.;
}


Float xyz_irradiance(
        const Float wv1, const Float wv2,
        Float (*get_xyz)(Float))
{
    Float wv = wv1; //int n = 1000;
    const Float dwv = Float(0.1);//(wv2-wv1)/Float(n);
    Float sum = 0;
    //for (int i=0; i<n; ++i)
    while (wv < wv2)
    {
        const Float wv_tmp = wv + dwv/Float(2.);// = (wv1 + i*dwv) + dwv/Float(2.);
        //const Float wv = (wv1 + i*dwv) + dwv/Float(2.);
        sum += get_xyz(wv_tmp) * Planck(wv_tmp*Float(1e-9)) * dwv;
        wv += dwv;
    }
    return sum * Float(1e-9);
}


Radiation_solver_shortwave::Radiation_solver_shortwave(
        const Gas_concs_gpu& gas_concs,
        const std::string& file_name_gas,
        const std::string& file_name_cloud,
        const std::string& file_name_aerosol)
{
    // Construct the gas optics classes for the solver.
    this->kdist_gpu = std::make_unique<Gas_optics_rrtmgp_rt>(
            load_and_init_gas_optics(gas_concs, file_name_gas));

    this->cloud_optics_gpu = std::make_unique<Cloud_optics_rt>(
            load_and_init_cloud_optics(file_name_cloud));

    this->aerosol_optics_gpu = std::make_unique<Aerosol_optics_rt>(
            load_and_init_aerosol_optics(file_name_aerosol));
}

void Radiation_solver_shortwave::load_mie_tables(
        const std::string& file_name_mie_bb,
        const std::string& file_name_mie_im,
        const bool switch_broadband,
        const bool switch_image)
{
    if (switch_broadband)
    {
        Netcdf_file mie_nc(file_name_mie_bb, Netcdf_mode::Read);
        const int n_bnd_sw = this->get_n_bnd_gpu();
        const int n_re  = mie_nc.get_dimension_size("r_eff");
        const int n_mie = mie_nc.get_dimension_size("n_ang");
        
        Array<Float,3> mie_cdf(mie_nc.get_variable<Float>("phase_cdf", {n_bnd_sw, n_mie}), {n_mie, 1, n_bnd_sw});
        Array<Float,4> mie_ang(mie_nc.get_variable<Float>("phase_cdf_angle", {n_bnd_sw, n_re, n_mie}), {n_mie, n_re, 1, n_bnd_sw});
        
        Array<Float,4> mie_phase(mie_nc.get_variable<Float>("phase", {n_bnd_sw, n_re, n_mie}), {n_mie, n_re, 1, n_bnd_sw});
        Array<Float,1> mie_phase_ang(mie_nc.get_variable<Float>("phase_angle", {n_mie}), {n_mie});

        this->mie_cdfs_bb = mie_cdf;
        this->mie_angs_bb = mie_ang;

        this->mie_phase_bb = mie_phase;
        this->mie_phase_angs_bb = mie_phase_ang;
    }
    
    if (switch_image)
    {
        Netcdf_file mie_nc(file_name_mie_im, Netcdf_mode::Read);
        const int n_bnd_sw = this->get_n_bnd_gpu();
        const int n_re  = mie_nc.get_dimension_size("r_eff");
        const int n_mie = mie_nc.get_dimension_size("n_ang");
        const int n_sub = mie_nc.get_dimension_size("sub_band");
        
        Array<Float,3> mie_cdf(mie_nc.get_variable<Float>("phase_cdf", {n_bnd_sw, n_sub, n_mie}), {n_mie, n_sub, n_bnd_sw});
        Array<Float,4> mie_ang(mie_nc.get_variable<Float>("phase_cdf_angle", {n_bnd_sw, n_sub, n_re, n_mie}), {n_mie, n_re, n_sub, n_bnd_sw});

        Array<Float,4> mie_phase(mie_nc.get_variable<Float>("phase", {n_bnd_sw, n_sub, n_re, n_mie}), {n_mie, n_re, n_sub, n_bnd_sw});
        Array<Float,1> mie_phase_ang(mie_nc.get_variable<Float>("phase_angle", {n_mie}), {n_mie});

        this->mie_cdfs = mie_cdf;
        this->mie_angs = mie_ang;

        this->mie_phase = mie_phase;
        this->mie_phase_angs = mie_phase_ang;
    
    }

}


void Radiation_solver_shortwave::solve_gpu(
        const bool tune_step,
        const bool switch_cloud_optics,
        const bool switch_cloud_mie,
        const bool switch_aerosol_optics,
        const bool switch_lu_albedo,
        const bool switch_delta_cloud,
        const bool switch_delta_aerosol,
        const bool switch_cloud_cam,
        const bool switch_raytracing,
        const Vector<int>& grid_cells,
        const Vector<Float>& grid_d,
        const Vector<int>& kn_grid,
        const int photons_per_pixel,
        const Gas_concs_gpu& gas_concs,
        const Array_gpu<Float,2>& p_lay, const Array_gpu<Float,2>& p_lev,
        const Array_gpu<Float,2>& t_lay, const Array_gpu<Float,2>& t_lev,
        const Array_gpu<Float,1>& z_lev,
        Array_gpu<Float,2>& col_dry,
        const Array_gpu<Float,2>& sfc_alb,
        const Array_gpu<Float,1>& tsi_scaling,
        const Array_gpu<Float,1>& mu0, const Array_gpu<Float,1>& azi,
        const Array_gpu<Float,2>& lwp, const Array_gpu<Float,2>& iwp,
        const Array_gpu<Float,2>& rel, const Array_gpu<Float,2>& rei,
        const Array_gpu<Float,1>& land_use_map,
        const Array_gpu<Float,2>& rh,
        const Aerosol_concs_gpu& aerosol_concs,
        const Camera& camera,
        Array_gpu<Float,3>& XYZ,
        Array_gpu<Float,2>& liwp_cam,
        Array_gpu<Float,2>& tauc_cam,
        Array_gpu<Float,2>& dist_cam,
        Array_gpu<Float,2>& zen_cam)

{
    const int n_col = p_lay.dim(1);
    const int n_lay = p_lay.dim(2);
    const int n_lev = p_lev.dim(2);
    const int n_gpt = this->kdist_gpu->get_ngpt();
    const int n_bnd = this->kdist_gpu->get_nband();

    const int n_mie = (switch_cloud_mie) ? this->mie_angs.dim(1) : 0;
    const int n_re = (switch_cloud_mie) ? this->mie_angs.dim(2) : 0;
    const int n_sub = (switch_cloud_mie) ? this->mie_angs.dim(3) : 3;

    const int cam_nx = XYZ.dim(1);
    const int cam_ny = XYZ.dim(2);
    const int cam_ns = XYZ.dim(3);
    const Bool top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

    optical_props = std::make_unique<Optical_props_2str_rt>(n_col, n_lay, *kdist_gpu);
    cloud_optical_props = std::make_unique<Optical_props_2str_rt>(n_col, n_lay, *cloud_optics_gpu);
    aerosol_optical_props = std::make_unique<Optical_props_2str_rt>(n_col, n_lay, *aerosol_optics_gpu);


    if (col_dry.size() == 0)
    {
        col_dry.set_dims({n_col, n_lay});
        Gas_optics_rrtmgp_rt::get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);
    }

    Array_gpu<Float,1> toa_src({n_col});
    Array_gpu<Float,2> flux_camera({cam_nx, cam_ny});

    Array<int,2> cld_mask_liq({n_col, n_lay});
    Array<int,2> cld_mask_ice({n_col, n_lay});

    Array_gpu<Float,3> mie_cdfs_sub;
    Array_gpu<Float,4> mie_angs_sub;
    Array_gpu<Float,4> mie_phase_sub;

    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(cam_ns, cam_nx, cam_ny, XYZ.ptr());

    const Array<int, 2>& band_limits_gpt(this->kdist_gpu->get_band_lims_gpoint());

    if (switch_raytracing)
    {
        int previous_band = 0;
        for (int igpt=1; igpt<=n_gpt; ++igpt)
        {
            int band = 0;
            for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
            {
                if (igpt <= band_limits_gpt({2, ibnd}))
                {
                    band = ibnd;
                    break;
                }
            }
            Array_gpu<Float,2> albedo;
            if (!switch_lu_albedo) albedo = sfc_alb.subset({{ {band, band}, {1, n_col}}});
    
            if (!tune_step && (! (band == 10 || band == 11 || band ==12))) continue;
    
            const Float solar_source_band = kdist_gpu->band_source(band_limits_gpt({1,band}), band_limits_gpt({2,band}));
    
            constexpr int n_col_block = 1<<14; // 2^14
    
            Array_gpu<Float,1> toa_src_temp({n_col_block});
            auto gas_optics_subset = [&](
                    const int col_s, const int n_col_subset)
            {
                // Run the gas_optics on a subset.
                kdist_gpu->gas_optics(
                        igpt,
                        col_s,
                        n_col_subset,
                        n_col,
                        p_lay,
                        p_lev,
                        t_lay,
                        gas_concs,
                        optical_props,
                        toa_src_temp,
                        col_dry);
            };
    
            const int n_blocks = n_col / n_col_block;
            const int n_col_residual = n_col % n_col_block;
    
            std::unique_ptr<Optical_props_arry_rt> optical_props_block =
                    std::make_unique<Optical_props_2str_rt>(n_col_block, n_lay, *kdist_gpu);
    
            for (int n=0; n<n_blocks; ++n)
            {
                const int col_s = n*n_col_block;
                gas_optics_subset(col_s, n_col_block);
            }
    
            if (n_col_residual > 0)
            {
                const int col_s = n_blocks*n_col_block;
                gas_optics_subset(col_s, n_col_residual);
            }
    
            if (tune_step) return;
    
            toa_src.fill(toa_src_temp({1}) * tsi_scaling({1}));
            if (switch_cloud_optics)
            {
                if (band > previous_band)
                {
                    cloud_optics_gpu->cloud_optics(
                            band,
                            lwp,
                            iwp,
                            rel,
                            rei,
                            *cloud_optical_props);
    
                    if (switch_delta_cloud)
                        cloud_optical_props->delta_scale();
                }
    
                // Add the cloud optical props to the gas optical properties.
                add_to(
                        dynamic_cast<Optical_props_2str_rt&>(*optical_props),
                        dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props));
            }
            else
            {
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, cloud_optical_props->get_tau().ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, cloud_optical_props->get_ssa().ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, cloud_optical_props->get_g().ptr());
            }
    
            if (switch_aerosol_optics)
            {
                if (band > previous_band)
                {
                    Aerosol_concs_gpu aerosol_concs_subset(aerosol_concs, 1, n_col);
                    aerosol_optics_gpu->aerosol_optics(
                            band,
                            aerosol_concs_subset,
                            rh, p_lev,
                            *aerosol_optical_props);
    
                    if (switch_delta_aerosol)
                        aerosol_optical_props->delta_scale();
                }
    
                // Add the aerosol optical props to the gas optical properties.
                add_to(
                        dynamic_cast<Optical_props_2str_rt&>(*optical_props),
                        dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props));
            }
            else
            {
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, aerosol_optical_props->get_tau().ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, aerosol_optical_props->get_ssa().ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, aerosol_optical_props->get_g().ptr());
            }
    
    
            /* rrtmgp's bands are quite broad, we divide each spectral band in three equally broad spectral intervals
               and run each g-point for each spectral interval, using the mean rayleigh scattering coefficient of each spectral interval
               in stead of RRTMGP's rayleigh scattering coefficients.
               The contribution of each spectral interval to the spectral band is based on the integrated (<>) Planck source function:
               <Planck(spectral interval)> / <Planck(spectral band)>, with a sun temperature of 5778 K. This is not entirely accurate because
               the sun is not a black body radiatior, but the approximations comes close enough.
    
               */
    
            // number of intervals
            const Array<Float, 2>& band_limits_wn(this->kdist_gpu->get_band_lims_wavenumber());
            const int nwv = n_sub;
    
            const Float wv1 = 1. / band_limits_wn({2,band}) * Float(1.e7);
            const Float wv2 = 1. / band_limits_wn({1,band}) * Float(1.e7);
            const Float dwv = (wv2-wv1)/Float(nwv);
    
            //
            const Float total_planck = Planck_integrator(wv1,wv2);
    
            for (int iwv=0; iwv<nwv; ++iwv)
            {
                const Float wv1_sub = wv1 + iwv*dwv;
                const Float wv2_sub = wv1 + (iwv+1)*dwv;
                const Float local_planck = Planck_integrator(wv1_sub,wv2_sub);
    
                // use RRTMGPs scattering coefficients if solving per band instead of subbands
                const Float rayleigh = (nwv==1) ? 0 : rayleigh_mean(wv1_sub, wv2_sub);
                const Float toa_factor = local_planck / total_planck * Float(1.)/solar_source_band;
    
                // XYZ factors
                Array<Float,1> xyz_factor({3});
                xyz_factor({1}) = xyz_irradiance(wv1_sub,wv2_sub,&get_x);
                xyz_factor({2}) = xyz_irradiance(wv1_sub,wv2_sub,&get_y);
                xyz_factor({3}) = xyz_irradiance(wv1_sub,wv2_sub,&get_z);
                Array_gpu<Float,1> xyz_factor_gpu(xyz_factor);
    
                if (switch_lu_albedo)
                {
                    if (albedo.size() == 0) albedo.set_dims({1, n_col});
                    spectral_albedo(n_col, wv1, wv2, land_use_map, albedo);
                }
    
                const Float zenith_angle = std::acos(mu0({1}));
                const Float azimuth_angle = azi({1});
    
                if (switch_cloud_mie)
                {
                    mie_cdfs_sub = mie_cdfs.subset({{ {1, n_mie}, {iwv+1,iwv+1}, {band, band} }});
                    mie_angs_sub = mie_angs.subset({{ {1, n_mie}, {1, n_re}, {iwv+1,iwv+1}, {band, band} }});
                    mie_phase_sub = mie_phase.subset({{ {1, n_mie}, {1, n_re}, {iwv+1,iwv+1}, {band, band} }});
                }
    
                raytracer.trace_rays(
                        igpt,
                        photons_per_pixel, n_lay,
                        grid_cells, grid_d, kn_grid,
                        z_lev,
                        mie_cdfs_sub,
                        mie_angs_sub,
                        mie_phase_sub,
                        mie_phase_angs,
                        rel,
                        dynamic_cast<Optical_props_2str_rt&>(*optical_props).get_tau(),
                        dynamic_cast<Optical_props_2str_rt&>(*optical_props).get_ssa(),
                        dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_tau(),
                        dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_ssa(),
                        dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_g(),
                        dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props).get_tau(),
                        dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props).get_ssa(),
                        dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props).get_g(),
                        albedo,
                        land_use_map,
                        zenith_angle,
                        azimuth_angle,
                        toa_src({1}),
                        toa_factor,
                        rayleigh,
                        col_dry,
                        gas_concs.get_vmr("h2o"),
                        camera,
                        flux_camera);
    
                raytracer.add_xyz_camera(
                        camera,
                        xyz_factor_gpu,
                        flux_camera,
                        XYZ);
    
            }
            previous_band = band;
        }
    }    
    
    if (switch_cloud_cam)
    {
        cloud_optics_gpu->cloud_optics(
                11, // 441-615 nm band
                lwp,
                iwp,
                rel,
                rei,
                *cloud_optical_props);
        
        raytracer.accumulate_clouds(
                grid_d,
                grid_cells,
                lwp,
                iwp,
                dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_tau(), 
                camera,
                liwp_cam,
                tauc_cam,
                dist_cam,
                zen_cam);
    }
}

void Radiation_solver_shortwave::solve_gpu_bb(
        const bool switch_cloud_optics,
        const bool switch_cloud_mie,
        const bool switch_aerosol_optics,
        const bool switch_lu_albedo,
        const bool switch_delta_cloud,
        const bool switch_delta_aerosol,
        const bool switch_cloud_cam,
        const bool switch_raytracing,
        const Vector<int>& grid_cells,
        const Vector<Float>& grid_d,
        const Vector<int>& kn_grid,
        const int photons_per_pixel,
        const Gas_concs_gpu& gas_concs,
        const Array_gpu<Float,2>& p_lay, const Array_gpu<Float,2>& p_lev,
        const Array_gpu<Float,2>& t_lay, const Array_gpu<Float,2>& t_lev,
        const Array_gpu<Float,1>& z_lev,
        Array_gpu<Float,2>& col_dry,
        const Array_gpu<Float,2>& sfc_alb,
        const Array_gpu<Float,1>& tsi_scaling,
        const Array_gpu<Float,1>& mu0, const Array_gpu<Float,1>& azi,
        const Array_gpu<Float,2>& lwp, const Array_gpu<Float,2>& iwp,
        const Array_gpu<Float,2>& rel, const Array_gpu<Float,2>& rei,
        const Array_gpu<Float,1>& land_use_map,
        const Array_gpu<Float,2>& rh,
        const Aerosol_concs_gpu& aerosol_concs,
        const Camera& camera,
        Array_gpu<Float,2>& radiance,
        Array_gpu<Float,2>& liwp_cam,
        Array_gpu<Float,2>& tauc_cam,
        Array_gpu<Float,2>& dist_cam,
        Array_gpu<Float,2>& zen_cam)
{
    const int n_col = p_lay.dim(1);
    const int n_lay = p_lay.dim(2);
    const int n_lev = p_lev.dim(2);
    const int n_gpt = this->kdist_gpu->get_ngpt();
    const int n_bnd = this->kdist_gpu->get_nband();

    const int n_mie = (switch_cloud_mie) ? this->mie_angs_bb.dim(1) : 0;
    const int n_re = (switch_cloud_mie) ? this->mie_angs_bb.dim(2) : 0;
    
    const int cam_nx = radiance.dim(1);
    const int cam_ny = radiance.dim(2);
    const Bool top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

    optical_props = std::make_unique<Optical_props_2str_rt>(n_col, n_lay, *kdist_gpu);
    cloud_optical_props = std::make_unique<Optical_props_2str_rt>(n_col, n_lay, *cloud_optics_gpu);
    aerosol_optical_props = std::make_unique<Optical_props_2str_rt>(n_col, n_lay, *aerosol_optics_gpu);

    if (col_dry.size() == 0)
    {
        col_dry.set_dims({n_col, n_lay});
        Gas_optics_rrtmgp_rt::get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);
    }

    Array_gpu<Float,1> toa_src({n_col});
    Array_gpu<Float,2> flux_camera({cam_nx, cam_ny});

    Array<int,2> cld_mask_liq({n_col, n_lay});
    Array<int,2> cld_mask_ice({n_col, n_lay});

    Array_gpu<Float,3> mie_cdfs_sub;
    Array_gpu<Float,4> mie_angs_sub;
    Array_gpu<Float,4> mie_phase_sub;

    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(cam_nx, cam_ny, radiance.ptr());

    const Array<int, 2>& band_limits_gpt(this->kdist_gpu->get_band_lims_gpoint());

    int previous_band = 0;
    if (switch_raytracing)
    {
        for (int igpt=1; igpt<=n_gpt; ++igpt)
        {
            int band = 0;
            for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
            {
                if (igpt <= band_limits_gpt({2, ibnd}))
                {
                    band = ibnd;
                    break;
                }
            }

            constexpr int n_col_block = 1<<14; // 2^14

            Array_gpu<Float,1> toa_src_temp({n_col_block});

            auto gas_optics_subset = [&](
                    const int col_s, const int n_col_subset)
            {
                // Run the gas_optics on a subset.
                kdist_gpu->gas_optics(
                        igpt,
                        col_s,
                        n_col_subset,
                        n_col,
                        p_lay,
                        p_lev,
                        t_lay,
                        gas_concs,
                        optical_props,
                        toa_src_temp,
                        col_dry);
            };

            const int n_blocks = n_col / n_col_block;
            const int n_col_residual = n_col % n_col_block;

            for (int n=0; n<n_blocks; ++n)
            {
                const int col_s = n*n_col_block;
                gas_optics_subset(col_s, n_col_block);
            }

            if (n_col_residual > 0)
            {
                const int col_s = n_blocks*n_col_block;
                gas_optics_subset(col_s, n_col_residual);
            }

            toa_src.fill(toa_src_temp({1}) * tsi_scaling({1}));

            if (switch_cloud_optics)
            {
                if (band > previous_band)
                {
                    cloud_optics_gpu->cloud_optics(
                            band,
                            lwp,
                            iwp,
                            rel,
                            rei,
                            *cloud_optical_props);

                    if (switch_delta_cloud)
                        cloud_optical_props->delta_scale();
                }
                // Add the cloud optical props to the gas optical properties.
                add_to(
                        dynamic_cast<Optical_props_2str_rt&>(*optical_props),
                        dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props));
            }
            else
            {
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, cloud_optical_props->get_tau().ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, cloud_optical_props->get_ssa().ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, cloud_optical_props->get_g().ptr());
            }

            if (switch_aerosol_optics)
            {
                if (band > previous_band)
                {
                    Aerosol_concs_gpu aerosol_concs_subset(aerosol_concs, 1, n_col);
                    aerosol_optics_gpu->aerosol_optics(
                            band,
                            aerosol_concs_subset,
                            rh, p_lev,
                            *aerosol_optical_props);

                    if (switch_delta_aerosol)
                        aerosol_optical_props->delta_scale();
                }

                // Add the aerosol optical props to the gas optical properties.
                add_to(
                        dynamic_cast<Optical_props_2str_rt&>(*optical_props),
                        dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props));
            }
            else
            {
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, aerosol_optical_props->get_tau().ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, aerosol_optical_props->get_ssa().ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_col, n_lay, aerosol_optical_props->get_g().ptr());
            }

            const Float zenith_angle = std::acos(mu0({1}));
            const Float azimuth_angle = azi({1});

            Array_gpu<Float,2> albedo;
            if (switch_lu_albedo)
            {
                albedo.set_dims({1, n_col});

                const Array<Float, 2>& band_limits_wn(this->kdist_gpu->get_band_lims_wavenumber());
                const Float wv1 = 1. / band_limits_wn({2,band}) * Float(1.e7);
                const Float wv2 = 1. / band_limits_wn({1,band}) * Float(1.e7);
                spectral_albedo(n_col, wv1, wv2, land_use_map, albedo);
            }
            else
            {
                albedo = sfc_alb.subset({{ {band, band}, {1, n_col}}});
            }

            if (switch_cloud_mie)
            {
                mie_cdfs_sub = mie_cdfs_bb.subset({{ {1, n_mie}, {1,1}, {band, band} }});
                mie_angs_sub = mie_angs_bb.subset({{ {1, n_mie}, {1, n_re}, {1,1}, {band, band} }});
                mie_phase_sub = mie_phase_bb.subset({{ {1, n_mie}, {1, n_re}, {1,1}, {band, band} }});
            }

            raytracer.trace_rays_bb(
                    igpt,
                    photons_per_pixel, n_lay,
                    grid_cells, grid_d, kn_grid,
                    z_lev,
                    mie_cdfs_sub,
                    mie_angs_sub,
                    mie_phase_sub,
                    mie_phase_angs_bb,
                    rel,
                    dynamic_cast<Optical_props_2str_rt&>(*optical_props).get_tau(),
                    dynamic_cast<Optical_props_2str_rt&>(*optical_props).get_ssa(),
                    dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_tau(),
                    dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_ssa(),
                    dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_g(),
                    dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props).get_tau(),
                    dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props).get_ssa(),
                    dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props).get_g(),
                    albedo,
                    land_use_map,
                    zenith_angle,
                    azimuth_angle,
                    toa_src({1}),
                    camera,
                    flux_camera);

            raytracer.add_camera(
                    camera,
                    flux_camera,
                    radiance);

            previous_band = band;
        }
    }

    if (switch_cloud_cam)
    {
        cloud_optics_gpu->cloud_optics(
                11, // 441-615 nm band
                lwp,
                iwp,
                rel,
                rei,
                *cloud_optical_props);
        
        raytracer.accumulate_clouds(
                grid_d,
                grid_cells,
                lwp,
                iwp,
                dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_tau(), 
                camera,
                liwp_cam,
                tauc_cam,
                dist_cam,
                zen_cam);
    }
}
