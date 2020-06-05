/*
 * This file is a stand-alone executable developed for the
 * testing of the C++ interface to the RTE+RRTMGP radiation code.
 *
 * It is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <chrono>

#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "Radiation_solver.h"


#ifdef FLOAT_SINGLE_RRTMGP
#define FLOAT_TYPE float
#else
#define FLOAT_TYPE double
#endif


template<typename TF>
void read_and_set_vmr(
        const std::string& gas_name, const int n_col, const int n_lay,
        const Netcdf_handle& input_nc, Gas_concs<TF>& gas_concs)
{
    const std::string vmr_gas_name = "vmr_" + gas_name;

    if (input_nc.variable_exists(vmr_gas_name))
    {
        std::map<std::string, int> dims = input_nc.get_variable_dimensions(vmr_gas_name);
        const int n_dims = dims.size();

        if (n_dims == 0)
        {
            gas_concs.set_vmr(gas_name, input_nc.get_variable<TF>(vmr_gas_name));
        }
        else if (n_dims == 1)
        {
            if (dims.at("lay") == n_lay)
                gas_concs.set_vmr(gas_name,
                        Array<TF,1>(input_nc.get_variable<TF>(vmr_gas_name, {n_lay}), {n_lay}));
            else
                throw std::runtime_error("Illegal dimensions of gas \"" + gas_name + "\" in input");
        }
        else if (n_dims == 2)
        {
            if (dims.at("lay") == n_lay && dims.at("col") == n_col)
                gas_concs.set_vmr(gas_name,
                        Array<TF,2>(input_nc.get_variable<TF>(vmr_gas_name, {n_lay, n_col}), {n_col, n_lay}));
            else
                throw std::runtime_error("Illegal dimensions of gas \"" + gas_name + "\" in input");
        }
    }
    else
    {
        Status::print_warning("Gas \"" + gas_name + "\" not available in input file.");
    }
}


template<typename TF>
void solve_radiation()
{
    ////// FLOW CONTROL SWITCHES //////
    const bool sw_output_optical = false;
    const bool sw_output_bnd_fluxes = false;


    ////// READ THE ATMOSPHERIC DATA //////
    Status::print_message("Reading atmospheric input data from NetCDF.");

    Netcdf_file input_nc("rte_rrtmgp_input.nc", Netcdf_mode::Read);

    const int n_col = input_nc.get_dimension_size("col");
    const int n_lay = input_nc.get_dimension_size("lay");
    const int n_lev = input_nc.get_dimension_size("lev");

    // Read the atmospheric fields.
    Array<TF,2> p_lay(input_nc.get_variable<TF>("p_lay", {n_lay, n_col}), {n_col, n_lay});
    Array<TF,2> t_lay(input_nc.get_variable<TF>("t_lay", {n_lay, n_col}), {n_col, n_lay});
    Array<TF,2> p_lev(input_nc.get_variable<TF>("p_lev", {n_lev, n_col}), {n_col, n_lev});
    Array<TF,2> t_lev(input_nc.get_variable<TF>("t_lev", {n_lev, n_col}), {n_col, n_lev});

    // Fetch the col_dry in case present.
    Array<TF,2> col_dry;
    if (input_nc.variable_exists("col_dry"))
    {
        col_dry.set_dims({n_col, n_lay});
        col_dry = std::move(input_nc.get_variable<TF>("col_dry", {n_lay, n_col}));
    }

    // Create container for the gas concentrations and read gases.
    Gas_concs<TF> gas_concs;

    read_and_set_vmr("h2o", n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("co2", n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("o3" , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("n2o", n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("co" , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("ch4", n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("o2" , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("n2" , n_col, n_lay, input_nc, gas_concs);

    read_and_set_vmr("ccl4"   , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc11"  , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc12"  , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc22"  , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc143a", n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc125" , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc23"  , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc32"  , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc134a", n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cf4"    , n_col, n_lay, input_nc, gas_concs);
    read_and_set_vmr("no2"    , n_col, n_lay, input_nc, gas_concs);


    ////// INITIALIZE THE SOLVER AND INIT K-DISTRIBUTION //////
    Status::print_message("Initializing the solvers.");
    Radiation_solver_longwave<TF> rad_lw(gas_concs, "coefficients_lw.nc");
    Radiation_solver_shortwave<TF> rad_sw(gas_concs, "coefficients_sw.nc");


    ////// READ THE SURFACE DATA //////
    // Read the boundary conditions for longwave.
    const int n_bnd_lw = rad_lw.get_n_bnd();
    const int n_gpt_lw = rad_lw.get_n_gpt();

    Array<TF,2> emis_sfc(input_nc.get_variable<TF>("emis_sfc", {n_col, n_bnd_lw}), {n_bnd_lw, n_col});
    Array<TF,1> t_sfc(input_nc.get_variable<TF>("t_sfc", {n_col}), {n_col});

    // Read the boundary conditions for shortwave.
    const int n_bnd_sw = rad_sw.get_n_bnd();
    const int n_gpt_sw = rad_sw.get_n_gpt();

    Array<TF,1> mu0(input_nc.get_variable<TF>("mu0", {n_col}), {n_col});
    Array<TF,2> sfc_alb_dir(input_nc.get_variable<TF>("sfc_alb_dir", {n_col, n_bnd_sw}), {n_bnd_sw, n_col});
    Array<TF,2> sfc_alb_dif(input_nc.get_variable<TF>("sfc_alb_dir", {n_col, n_bnd_sw}), {n_bnd_sw, n_col});

    Array<TF,1> tsi_scaling({n_col});
    if (input_nc.variable_exists("tsi"))
    {
        Array<TF,1> tsi(input_nc.get_variable<TF>("tsi", {n_col}), {n_col});
        const TF tsi_ref = rad_sw.get_tsi();
        for (int icol=1; icol<=n_col; ++icol)
            tsi_scaling({icol}) = tsi({icol}) / tsi_ref;
    }
    else
        for (int icol=1; icol<=n_col; ++icol)
            tsi_scaling({icol}) = TF(1.);


    ////// CREATE THE OUTPUT ARRAYS //////
    Array<TF,3> lw_tau;
    Array<TF,3> lay_source;
    Array<TF,3> lev_source_inc;
    Array<TF,3> lev_source_dec;
    Array<TF,2> sfc_source;

    Array<TF,3> sw_tau;
    Array<TF,3> ssa;
    Array<TF,3> g;
    Array<TF,2> toa_source;

    if (sw_output_optical)
    {
        lw_tau        .set_dims({n_col, n_lay, n_gpt_lw});
        lay_source    .set_dims({n_col, n_lay, n_gpt_lw});
        lev_source_inc.set_dims({n_col, n_lay, n_gpt_lw});
        lev_source_dec.set_dims({n_col, n_lay, n_gpt_lw});
        sfc_source    .set_dims({n_col, n_gpt_lw});

        sw_tau    .set_dims({n_col, n_lay, n_gpt_sw});
        ssa       .set_dims({n_col, n_lay, n_gpt_sw});
        g         .set_dims({n_col, n_lay, n_gpt_sw});
        toa_source.set_dims({n_col, n_gpt_sw});
    }

    Array<TF,2> lw_flux_up ({n_col, n_lev});
    Array<TF,2> lw_flux_dn ({n_col, n_lev});
    Array<TF,2> lw_flux_net({n_col, n_lev});

    Array<TF,3> lw_bnd_flux_up;
    Array<TF,3> lw_bnd_flux_dn;
    Array<TF,3> lw_bnd_flux_net;

    if (sw_output_bnd_fluxes)
    {
        lw_bnd_flux_up .set_dims({n_col, n_lev, n_bnd_lw});
        lw_bnd_flux_dn .set_dims({n_col, n_lev, n_bnd_lw});
        lw_bnd_flux_net.set_dims({n_col, n_lev, n_bnd_lw});
    }

    Array<TF,2> sw_flux_up    ({n_col, n_lev});
    Array<TF,2> sw_flux_dn    ({n_col, n_lev});
    Array<TF,2> sw_flux_dn_dir({n_col, n_lev});
    Array<TF,2> sw_flux_net   ({n_col, n_lev});

    Array<TF,3> sw_bnd_flux_up;
    Array<TF,3> sw_bnd_flux_dn;
    Array<TF,3> sw_bnd_flux_dn_dir;
    Array<TF,3> sw_bnd_flux_net;

    if (sw_output_bnd_fluxes)
    {
        sw_bnd_flux_up    .set_dims({n_col, n_lev, n_bnd_sw});
        sw_bnd_flux_dn    .set_dims({n_col, n_lev, n_bnd_sw});
        sw_bnd_flux_dn_dir.set_dims({n_col, n_lev, n_bnd_sw});
        sw_bnd_flux_net   .set_dims({n_col, n_lev, n_bnd_sw});
    }


    ////// SOLVE THE RADIATION //////
    Status::print_message("Solving the longwave radiation.");

    auto time_start = std::chrono::high_resolution_clock::now();

    rad_lw.solve(
            sw_output_optical,
            sw_output_bnd_fluxes,
            gas_concs,
            p_lay, p_lev,
            t_lay, t_lev,
            col_dry,
            t_sfc, emis_sfc,
            lw_tau, lay_source, lev_source_inc, lev_source_dec, sfc_source,
            lw_flux_up, lw_flux_dn, lw_flux_net,
            lw_bnd_flux_up, lw_bnd_flux_dn, lw_bnd_flux_net);

    auto time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(time_end-time_start).count();

    Status::print_message("Duration: " + std::to_string(duration) + " (ms)");

    // Solving the shortwave radiation
    Status::print_message("Solving the shortwave radiation.");

    time_start = std::chrono::high_resolution_clock::now();

    rad_sw.solve(
            sw_output_optical,
            sw_output_bnd_fluxes,
            gas_concs,
            p_lay, p_lev,
            t_lay, t_lev,
            col_dry,
            sfc_alb_dir, sfc_alb_dif,
            tsi_scaling, mu0,
            sw_tau, ssa, g,
            toa_source,
            sw_flux_up, sw_flux_dn,
            sw_flux_dn_dir, sw_flux_net,
            sw_bnd_flux_up, sw_bnd_flux_dn,
            sw_bnd_flux_dn_dir, sw_bnd_flux_net);

    time_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(time_end-time_start).count();

    Status::print_message("Duration: " + std::to_string(duration) + " (ms)");

    ////// SAVING THE OUTPUT TO NETCDF //////
    Status::print_message("Saving the output to NetCDF.");

    Netcdf_file output_nc("rte_rrtmgp_output.nc", Netcdf_mode::Create);
    output_nc.add_dimension("col", n_col);
    output_nc.add_dimension("lay", n_lay);
    output_nc.add_dimension("lev", n_lev);
    output_nc.add_dimension("pair", 2);

    auto nc_lay = output_nc.add_variable<TF>("p_lay", {"lay", "col"});
    auto nc_lev = output_nc.add_variable<TF>("p_lev", {"lev", "col"});

    nc_lay.insert(p_lay.v(), {0, 0});
    nc_lev.insert(p_lev.v(), {0, 0});

    output_nc.add_dimension("gpt_lw", n_gpt_lw);
    output_nc.add_dimension("band_lw", n_bnd_lw);

    output_nc.add_dimension("gpt_sw", n_gpt_sw);
    output_nc.add_dimension("band_sw", n_bnd_sw);

    auto nc_lw_band_lims_wvn = output_nc.add_variable<TF>("lw_band_lims_wvn", {"band_lw", "pair"});
    nc_lw_band_lims_wvn.insert(rad_lw.get_band_lims_wavenumber().v(), {0, 0});

    auto nc_sw_band_lims_wvn = output_nc.add_variable<TF>("sw_band_lims_wvn", {"band_sw", "pair"});
    nc_sw_band_lims_wvn.insert(rad_sw.get_band_lims_wavenumber().v(), {0, 0});

    if (sw_output_optical)
    {
        auto nc_lw_band_lims_gpt = output_nc.add_variable<int>("lw_band_lims_gpt", {"band_lw", "pair"});
        nc_lw_band_lims_gpt.insert(rad_lw.get_band_lims_gpoint().v(), {0, 0});

        auto nc_lw_tau = output_nc.add_variable<TF>("lw_tau", {"gpt_lw", "lay", "col"});
        nc_lw_tau.insert(lw_tau.v(), {0, 0, 0});

        // Second, store the sources.
        auto nc_lay_source     = output_nc.add_variable<TF>("lay_source"    , {"gpt_lw", "lay", "col"});
        auto nc_lev_source_inc = output_nc.add_variable<TF>("lev_source_inc", {"gpt_lw", "lay", "col"});
        auto nc_lev_source_dec = output_nc.add_variable<TF>("lev_source_dec", {"gpt_lw", "lay", "col"});

        auto nc_sfc_source = output_nc.add_variable<TF>("sfc_source", {"gpt_lw", "col"});

        nc_lay_source.insert    (lay_source.v()    , {0, 0, 0});
        nc_lev_source_inc.insert(lev_source_inc.v(), {0, 0, 0});
        nc_lev_source_dec.insert(lev_source_dec.v(), {0, 0, 0});

        nc_sfc_source.insert(sfc_source.v(), {0, 0});

        auto nc_sw_band_lims_gpt = output_nc.add_variable<int>("sw_band_lims_gpt", {"band_sw", "pair"});
        nc_sw_band_lims_gpt.insert(rad_sw.get_band_lims_gpoint().v(), {0, 0});

        auto nc_sw_tau = output_nc.add_variable<TF>("sw_tau", {"gpt_sw", "lay", "col"});
        auto nc_ssa    = output_nc.add_variable<TF>("ssa"   , {"gpt_sw", "lay", "col"});
        auto nc_g      = output_nc.add_variable<TF>("g"     , {"gpt_sw", "lay", "col"});

        nc_sw_tau.insert(sw_tau.v(), {0, 0, 0});
        nc_ssa   .insert(ssa   .v(), {0, 0, 0});
        nc_g     .insert(g     .v(), {0, 0, 0});

        auto nc_toa_source = output_nc.add_variable<TF>("toa_source", {"gpt_sw", "col"});
        nc_toa_source.insert(toa_source.v(), {0, 0});
    }

    // Save the output of the flux calculation to disk.
    auto nc_lw_flux_up  = output_nc.add_variable<TF>("lw_flux_up" , {"lev", "col"});
    auto nc_lw_flux_dn  = output_nc.add_variable<TF>("lw_flux_dn" , {"lev", "col"});
    auto nc_lw_flux_net = output_nc.add_variable<TF>("lw_flux_net", {"lev", "col"});

    nc_lw_flux_up .insert(lw_flux_up .v(), {0, 0});
    nc_lw_flux_dn .insert(lw_flux_dn .v(), {0, 0});
    nc_lw_flux_net.insert(lw_flux_net.v(), {0, 0});

    if (sw_output_bnd_fluxes)
    {
        auto nc_lw_bnd_flux_up  = output_nc.add_variable<TF>("lw_bnd_flux_up" , {"band_lw", "lev", "col"});
        auto nc_lw_bnd_flux_dn  = output_nc.add_variable<TF>("lw_bnd_flux_dn" , {"band_lw", "lev", "col"});
        auto nc_lw_bnd_flux_net = output_nc.add_variable<TF>("lw_bnd_flux_net", {"band_lw", "lev", "col"});

        nc_lw_bnd_flux_up .insert(lw_bnd_flux_up .v(), {0, 0, 0});
        nc_lw_bnd_flux_dn .insert(lw_bnd_flux_dn .v(), {0, 0, 0});
        nc_lw_bnd_flux_net.insert(lw_bnd_flux_net.v(), {0, 0, 0});
    }

    auto nc_sw_flux_up     = output_nc.add_variable<TF>("sw_flux_up"    , {"lev", "col"});
    auto nc_sw_flux_dn     = output_nc.add_variable<TF>("sw_flux_dn"    , {"lev", "col"});
    auto nc_sw_flux_dn_dir = output_nc.add_variable<TF>("sw_flux_dn_dir", {"lev", "col"});
    auto nc_sw_flux_net    = output_nc.add_variable<TF>("sw_flux_net"   , {"lev", "col"});

    nc_sw_flux_up    .insert(sw_flux_up    .v(), {0, 0});
    nc_sw_flux_dn    .insert(sw_flux_dn    .v(), {0, 0});
    nc_sw_flux_dn_dir.insert(sw_flux_dn_dir.v(), {0, 0});
    nc_sw_flux_net   .insert(sw_flux_net   .v(), {0, 0});

    if (sw_output_bnd_fluxes)
    {
        auto nc_sw_bnd_flux_up     = output_nc.add_variable<TF>("sw_bnd_flux_up"    , {"band_sw", "lev", "col"});
        auto nc_sw_bnd_flux_dn     = output_nc.add_variable<TF>("sw_bnd_flux_dn"    , {"band_sw", "lev", "col"});
        auto nc_sw_bnd_flux_dn_dir = output_nc.add_variable<TF>("sw_bnd_flux_dn_dir", {"band_sw", "lev", "col"});
        auto nc_sw_bnd_flux_net    = output_nc.add_variable<TF>("sw_bnd_flux_net"   , {"band_sw", "lev", "col"});

        nc_sw_bnd_flux_up    .insert(sw_bnd_flux_up    .v(), {0, 0, 0});
        nc_sw_bnd_flux_dn    .insert(sw_bnd_flux_dn    .v(), {0, 0, 0});
        nc_sw_bnd_flux_dn_dir.insert(sw_bnd_flux_dn_dir.v(), {0, 0, 0});
        nc_sw_bnd_flux_net   .insert(sw_bnd_flux_net   .v(), {0, 0, 0});
    }

    Status::print_message("Finished.");
}

int main()
{
    try
    {
        solve_radiation<FLOAT_TYPE>();
    }

    // Catch any exceptions and return 1.
    catch (const std::exception& e)
    {
        std::string error = "EXCEPTION: " + std::string(e.what());
        Status::print_message(error);
        return 1;
    }
    catch (...)
    {
        Status::print_message("UNHANDLED EXCEPTION!");
        return 1;
    }

    // Return 0 in case of normal exit.
    return 0;
}
