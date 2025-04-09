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

#include "Radiation_solver_rt.h"
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
        __global__
        void scale_tau_kernel(Float* tau, const int ncol, const int nlay, Float scale_factor) {
                const int icol = blockIdx.x*blockDim.x + threadIdx.x;
                const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

                if ( (icol < ncol) && (ilay < nlay) )
                {
                const int idx = icol + ilay*ncol;
                tau[idx] = tau[idx] * scale_factor;
                }
        }    

        void scale_tau(Float* tau, const int ncol, const int nlay, Float scale_factor) {
                const int block_col = 64;
                const int block_lay = 1;
                const int grid_col  = ncol/block_col + (ncol%block_col > 0);
                const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

                dim3 grid_gpu(grid_col, grid_lay);
                dim3 block_gpu(block_col, block_lay);
                scale_tau_kernel<<<grid_gpu, block_gpu>>>(tau, ncol, nlay, scale_factor);
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

        Float diamice_lwr = coef_nc.get_variable<Float>("diamice_lwr");
        Float diamice_upr = coef_nc.get_variable<Float>("diamice_upr");
        Float diamice_fac = coef_nc.get_variable<Float>("diamice_fac");

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
                diamice_lwr, diamice_upr, diamice_fac,
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
        const bool switch_single_gpt,
        const int single_gpt,
        const Gas_concs_gpu& gas_concs,
        const Array_gpu<Float,2>& p_lay, const Array_gpu<Float,2>& p_lev,
        const Array_gpu<Float,2>& t_lay, const Array_gpu<Float,2>& t_lev,
        Array_gpu<Float,2>& col_dry,
        const Array_gpu<Float,1>& t_sfc, const Array_gpu<Float,2>& emis_sfc,
        const Array_gpu<Float,2>& lwp, const Array_gpu<Float,2>& iwp,
        const Array_gpu<Float,2>& rel, const Array_gpu<Float,2>& dei,
        Array_gpu<Float,2>& tau, Array_gpu<Float,2>& lay_source,
        Array_gpu<Float,2>& lev_source_inc, Array_gpu<Float,2>& lev_source_dec, Array_gpu<Float,1>& sfc_source,
        Array_gpu<Float,2>& lw_flux_up, Array_gpu<Float,2>& lw_flux_dn, Array_gpu<Float,2>& lw_flux_net,
        Array_gpu<Float,2>& lw_gpt_flux_up, Array_gpu<Float,2>& lw_gpt_flux_dn, Array_gpu<Float,2>& lw_gpt_flux_net)
{

    throw std::runtime_error("Longwave raytracing is not implemented");

    /*
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
                    dei,
                    *cloud_optical_props);
            // cloud->delta_scale();

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_1scl_rt&>(*optical_props),
                    dynamic_cast<Optical_props_1scl_rt&>(*cloud_optical_props));
        }

        // Store the optical properties, if desired.
        if (switch_single_gpt && igpt == single_gpt)
        {
            lay_source = (*sources).get_lay_source();
            lev_source_inc = (*sources).get_lev_source_inc();
            lev_source_dec = (*sources).get_lev_source_dec();
            sfc_source = (*sources).get_sfc_source();


            //gpt_combine_kernel_launcher_cuda_rt::get_from_gpoint(
            //        n_col, n_lay, igpt-1, tau.ptr(), lay_source.ptr(), lev_source_inc.ptr(), lev_source_dec.ptr(),
            //        optical_props->get_tau().ptr(), (*sources).get_lay_source().ptr(),
            //        (*sources).get_lev_source_inc().ptr(), (*sources).get_lev_source_dec().ptr());

            //gpt_combine_kernel_launcher_cuda_rt::get_from_gpoint(
            //        n_col, igpt-1, sfc_source.ptr(), (*sources).get_sfc_source().ptr());
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

            if (switch_single_gpt && igpt == single_gpt)
            {
                lw_gpt_flux_up = (*fluxes).get_flux_up();
                lw_gpt_flux_dn = (*fluxes).get_flux_dn();
                lw_gpt_flux_net = (*fluxes).get_flux_net();
                //gpt_combine_kernel_launcher_cuda_rt::get_from_gpoint(
                //        n_col, n_lev, igpt-1, lw_gpt_flux_up.ptr(), lw_gpt_flux_dn.ptr(), lw_gpt_flux_net.ptr(),
                //        (*fluxes).get_flux_up().ptr(), (*fluxes).get_flux_dn().ptr(), (*fluxes).get_flux_net().ptr());
            }
        }
    }
    */
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
        const std::string& file_name_mie)
{
    Netcdf_file mie_nc(file_name_mie, Netcdf_mode::Read);
    const int n_re  = mie_nc.get_dimension_size("r_eff");
    const int n_mie = mie_nc.get_dimension_size("n_ang_cdf");

    const int n_bnd_sw = this->get_n_bnd_gpu();
    Array<Float,2> mie_cdf(mie_nc.get_variable<Float>("phase_cdf", {n_bnd_sw, n_mie}), {n_mie, n_bnd_sw});
    Array<Float,3> mie_ang(mie_nc.get_variable<Float>("phase_cdf_angle", {n_bnd_sw, n_re, n_mie}), {n_mie, n_re, n_bnd_sw});

    this->mie_cdfs = mie_cdf;
    this->mie_angs = mie_ang;

}

void Radiation_solver_shortwave::solve_gpu(
        const bool switch_fluxes,
        const bool switch_twostream,
        const bool switch_raytracing,
        const bool switch_independent_column,
        const bool switch_cloud_optics,
        const bool switch_cloud_mie,
        const bool switch_aerosol_optics,
        const bool switch_single_gpt,
        const bool switch_delta_cloud,
        const bool switch_delta_aerosol,
        const bool switch_attenuate_tica,
        const int single_gpt,
        const Int ray_count,
        const Vector<int> grid_cells,
        const Vector<Float> grid_d,
        const Vector<int> kn_grid,
        const Gas_concs_gpu& gas_concs,
        const Array_gpu<Float,2>& p_lay, const Array_gpu<Float,2>& p_lev,
        const Array_gpu<Float,2>& t_lay, const Array_gpu<Float,2>& t_lev,
        Array_gpu<Float,2>& col_dry,
        const Array_gpu<Float,2>& sfc_alb_dir, const Array_gpu<Float,2>& sfc_alb_dif,
        const Array_gpu<Float,1>& tsi_scaling,
        const Array_gpu<Float,1>& mu0, const Array_gpu<Float,1>& azi,
        const Array_gpu<Float,2>& lwp, const Array_gpu<Float,2>& iwp,
        const Array_gpu<Float,2>& rel, const Array_gpu<Float,2>& dei,
        const Array_gpu<Float,2>& rh,
        const Aerosol_concs_gpu& aerosol_concs,
        Array_gpu<Float,2>& tot_tau_out, Array_gpu<Float,2>& tot_ssa_out,
        Array_gpu<Float,2>& cld_tau_out, Array_gpu<Float,2>& cld_ssa_out, Array_gpu<Float,2>& cld_asy_out,
        Array_gpu<Float,2>& aer_tau_out, Array_gpu<Float,2>& aer_ssa_out, Array_gpu<Float,2>& aer_asy_out,
        Array_gpu<Float,2>& sw_flux_up, Array_gpu<Float,2>& sw_flux_dn,
        Array_gpu<Float,2>& sw_flux_dn_dir, Array_gpu<Float,2>& sw_flux_net,
        Array_gpu<Float,2>& sw_gpt_flux_up, Array_gpu<Float,2>& sw_gpt_flux_dn,
        Array_gpu<Float,2>& sw_gpt_flux_dn_dir, Array_gpu<Float,2>& sw_gpt_flux_net,
        Array_gpu<Float,2>& rt_flux_tod_up,
        Array_gpu<Float,2>& rt_flux_sfc_dir,
        Array_gpu<Float,2>& rt_flux_sfc_dif,
        Array_gpu<Float,2>& rt_flux_sfc_up,
        Array_gpu<Float,3>& rt_flux_abs_dir,
        Array_gpu<Float,3>& rt_flux_abs_dif)
{
    const int n_col = p_lay.dim(1);
    const int n_lay = p_lay.dim(2);
    const int n_lev = p_lev.dim(2);
    const int n_gpt = this->kdist_gpu->get_ngpt();
    const int n_bnd = this->kdist_gpu->get_nband();

    const int n_mie = (switch_cloud_mie) ? this->mie_angs.dim(1) : 0;
    const int n_re = (switch_cloud_mie) ? this->mie_angs.dim(2) : 0;

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

    Array<int,2> cld_mask_liq({n_col, n_lay});
    Array<int,2> cld_mask_ice({n_col, n_lay});

    Array_gpu<Float,2> mie_cdfs_sub;
    Array_gpu<Float,3> mie_angs_sub;

    if (switch_fluxes)
    {
        if (switch_twostream)
        {
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_lev, grid_cells.y, grid_cells.x, sw_flux_up.ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_lev, grid_cells.y, grid_cells.x, sw_flux_dn.ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_lev, grid_cells.y, grid_cells.x, sw_flux_dn_dir.ptr());
                Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(n_lev, grid_cells.y, grid_cells.x, sw_flux_net.ptr());
        }
        if (switch_raytracing)
        {
            Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.y, grid_cells.x, rt_flux_tod_up.ptr());
            Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.y, grid_cells.x, rt_flux_sfc_dir.ptr());
            Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.y, grid_cells.x, rt_flux_sfc_dif.ptr());
            Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.y, grid_cells.x, rt_flux_sfc_up.ptr());
            Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.z, grid_cells.y, grid_cells.x, rt_flux_abs_dir.ptr());
            Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.z, grid_cells.y, grid_cells.x, rt_flux_abs_dif.ptr());
        }
    }

    const Array<int, 2>& band_limits_gpt(this->kdist_gpu->get_band_lims_gpoint());
    int previous_band = 0;
    Float attenuate_scale_factor = 1/tsi_scaling({1});

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

        // We loop over the gas optics, due to memory constraints
        constexpr int n_col_block = 1<<14;

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

        if (n_blocks > 0)
        {
            for (int n=0; n<n_blocks; ++n)
            {
                const int col_s = n*n_col_block;
                gas_optics_subset(col_s, n_col_block);
            }
        }

        if (n_col_residual > 0)
        {
            const int col_s = n_blocks*n_col_block;
            gas_optics_subset(col_s, n_col_residual);
        }

        toa_src.fill(toa_src_temp({1}) * tsi_scaling({1}));

        if (switch_attenuate_tica)
        {
                scale_tau(dynamic_cast<Optical_props_2str_rt&>(*optical_props).get_tau().ptr(), n_col, n_lay, attenuate_scale_factor);
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
                if (switch_attenuate_tica)
                {
                        scale_tau(dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props).get_tau().ptr(), n_col, n_lay, attenuate_scale_factor);
                }
            }

            // Add the cloud optical props to the gas optical properties.
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

        if (switch_cloud_optics)
        {
            if (band > previous_band)
            {
                cloud_optics_gpu->cloud_optics(
                        band,
                        lwp,
                        iwp,
                        rel,
                        dei,
                        *cloud_optical_props);

                if (switch_delta_cloud)
                    cloud_optical_props->delta_scale();
                if (switch_attenuate_tica)
                {
                        scale_tau(dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_tau().ptr(), n_col, n_lay, attenuate_scale_factor);
                }
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

        // Store the optical properties, if desired
        if (switch_single_gpt && igpt == single_gpt)
        {
            tot_tau_out = optical_props->get_tau();
            tot_ssa_out = optical_props->get_ssa();
            cld_tau_out = cloud_optical_props->get_tau();
            cld_ssa_out = cloud_optical_props->get_ssa();
            cld_asy_out = cloud_optical_props->get_g();
            aer_tau_out = aerosol_optical_props->get_tau();
            aer_ssa_out = aerosol_optical_props->get_ssa();
            aer_asy_out = aerosol_optical_props->get_g();
        }

        if (switch_fluxes)
        {
            std::unique_ptr<Fluxes_broadband_rt> fluxes =
                    std::make_unique<Fluxes_broadband_rt>(grid_cells.x, grid_cells.y, grid_cells.z, n_lev);

            if (switch_twostream)
            {

                rte_sw.rte_sw(
                        optical_props,
                        top_at_1,
                        mu0,
                        toa_src,
                        sfc_alb_dir.subset({{ {band, band}, {1, n_col}}}),
                        sfc_alb_dif.subset({{ {band, band}, {1, n_col}}}),
                        Array_gpu<Float,1>(), // Add an empty array, no inc_flux.
                        (*fluxes).get_flux_up(),
                        (*fluxes).get_flux_dn(),
                        (*fluxes).get_flux_dn_dir());
            }

            if (switch_raytracing)
            {
                Float zenith_angle = std::acos(mu0({1}));
                Float azimuth_angle = azi({1}); // sun approximately from south

                if (switch_cloud_mie)
                {
                    mie_cdfs_sub = mie_cdfs.subset({{ {1, n_mie}, {band, band} }});
                    mie_angs_sub = mie_angs.subset({{ {1, n_mie}, {1, n_re}, {band, band} }});
                }

                raytracer.trace_rays(
                        igpt,
                        switch_independent_column,
                        ray_count,
                        grid_cells,
                        grid_d,
                        kn_grid,
                        mie_cdfs_sub,
                        mie_angs_sub,
                        dynamic_cast<Optical_props_2str_rt&>(*optical_props).get_tau(),
                        dynamic_cast<Optical_props_2str_rt&>(*optical_props).get_ssa(),
                        dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_tau(),
                        dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_ssa(),
                        dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_g(),
                        dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props).get_tau(),
                        dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props).get_ssa(),
                        dynamic_cast<Optical_props_2str_rt&>(*aerosol_optical_props).get_g(),
                        rel,
                        sfc_alb_dir.subset({{ {band, band}, {1, n_col}}}),
                        zenith_angle,
                        azimuth_angle,
                        toa_src({1}) * mu0({1}),
                        Float(0.),
                        (*fluxes).get_flux_tod_dn(),
                        (*fluxes).get_flux_tod_up(),
                        (*fluxes).get_flux_sfc_dir(),
                        (*fluxes).get_flux_sfc_dif(),
                        (*fluxes).get_flux_sfc_up(),
                        (*fluxes).get_flux_abs_dir(),
                        (*fluxes).get_flux_abs_dif());
            }

            (*fluxes).net_flux();
            if (switch_twostream)
            {
                Gpt_combine_kernels_cuda_rt::add_from_gpoint(
                        n_col, n_lev, sw_flux_up.ptr(), sw_flux_dn.ptr(), sw_flux_dn_dir.ptr(), sw_flux_net.ptr(),
                        (*fluxes).get_flux_up().ptr(), (*fluxes).get_flux_dn().ptr(), (*fluxes).get_flux_dn_dir().ptr(), (*fluxes).get_flux_net().ptr());
            }

            if (switch_raytracing)
            {
                Gpt_combine_kernels_cuda_rt::add_from_gpoint(
                        grid_cells.x, grid_cells.y, rt_flux_tod_up.ptr(), rt_flux_sfc_dir.ptr(), rt_flux_sfc_dif.ptr(), rt_flux_sfc_up.ptr(),
                        (*fluxes).get_flux_tod_up().ptr(), (*fluxes).get_flux_sfc_dir().ptr(), (*fluxes).get_flux_sfc_dif().ptr(), (*fluxes).get_flux_sfc_up().ptr());

                Gpt_combine_kernels_cuda_rt::add_from_gpoint(
                        n_col, grid_cells.z, rt_flux_abs_dir.ptr(), rt_flux_abs_dif.ptr(),
                        (*fluxes).get_flux_abs_dir().ptr(), (*fluxes).get_flux_abs_dif().ptr());
            }

            if (switch_single_gpt && igpt == single_gpt)
            {
                sw_gpt_flux_up = (*fluxes).get_flux_up();
                sw_gpt_flux_dn = (*fluxes).get_flux_dn();
                sw_gpt_flux_dn_dir = (*fluxes).get_flux_dn_dir();
                sw_gpt_flux_net = (*fluxes).get_flux_net();
            }
        }
        previous_band = band;
    }
}
