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

#include <boost/algorithm/string.hpp>
#include <chrono>
#include <iomanip>

#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "Radiation_solver.h"
#include "tilted_column.h"

#ifdef FLOAT_SINGLE_RRTMGP
#define FLOAT_TYPE float
#else
#define FLOAT_TYPE double
#endif


template<typename TF>
void read_and_set_vmr(
        const std::string& gas_name, const int n_y, const int n_x, const int n_col, const int n_lay,
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
        	throw std::runtime_error("only 0D, 1D or 3D allowed in input");
        }
        else if (n_dims == 3)
        {
            if (dims.at("lay") == n_lay && dims.at("x")*dims.at("y") == n_col)
                gas_concs.set_vmr(gas_name,
                        Array<TF,2>(input_nc.get_variable<TF>(vmr_gas_name, {n_lay, n_y, n_x}), {n_col, n_lay}));
            else
                throw std::runtime_error("Illegal dimensions of gas \"" + gas_name + "\" in input");
        }
    }
    else
    {
        Status::print_warning("Gas \"" + gas_name + "\" not available in input file.");
    }
}


bool parse_command_line_options(
        std::map<std::string, std::pair<bool, std::string>>& command_line_options,
        int argc, char** argv)
{
    for (int i=1; i<argc; ++i)
    {
        std::string argument(argv[i]);
        boost::trim(argument);

        if (argument == "-h" || argument == "--help")
        {
            Status::print_message("Possible usage:");
            for (const auto& clo : command_line_options)
            {
                std::ostringstream ss;
                ss << std::left << std::setw(30) << ("--" + clo.first);
                ss << clo.second.second << std::endl;
                Status::print_message(ss);
            }
            return true;
        }

        // Check if option starts with --
        if (argument[0] != '-' || argument[1] != '-')
        {
            std::string error = argument + " is an illegal command line option.";
            throw std::runtime_error(error);
        }
        else
            argument.erase(0, 2);

        // Check if option has prefix no-
        bool enable = true;
        if (argument[0] == 'n' && argument[1] == 'o' && argument[2] == '-')
        {
            enable = false;
            argument.erase(0, 3);
        }

        if (command_line_options.find(argument) == command_line_options.end())
        {
            std::string error = argument + " is an illegal command line option.";
            throw std::runtime_error(error);
        }
        else
            command_line_options.at(argument).first = enable;
    }

    return false;
}


void print_command_line_options(
        const std::map<std::string, std::pair<bool, std::string>>& command_line_options)
{
    Status::print_message("Solver settings:");
    for (const auto& option : command_line_options)
    {
        std::ostringstream ss;
        ss << std::left << std::setw(20) << (option.first);
        ss << " = " << std::boolalpha << option.second.first << std::endl;
        Status::print_message(ss);
    }
}


template<typename TF>
void solve_radiation(int argc, char** argv)
{
    Status::print_message("###### Starting RTE+RRTMGP solver ######");

    ////// FLOW CONTROL SWITCHES //////
    // Parse the command line options.
    std::map<std::string, std::pair<bool, std::string>> command_line_options {
        {"tilting"          , { false, "Computed tilted columns of 3D input."}},
        {"shortwave"        , { true,  "Enable computation of shortwave radiation."}},
        {"longwave"         , { true,  "Enable computation of longwave radiation." }},
        {"fluxes"           , { true,  "Enable computation of fluxes."             }},
        {"cloud-optics"     , { false, "Enable cloud optics."                      }},
        {"output-optical"   , { false, "Enable output of optical properties."      }},
        {"output-bnd-fluxes", { false, "Enable output of band fluxes."             }} };

    if (parse_command_line_options(command_line_options, argc, argv))
        return;

    const bool switch_tilting           = command_line_options.at("tilting"          ).first;
    const bool switch_shortwave         = command_line_options.at("shortwave"        ).first;
    const bool switch_longwave          = command_line_options.at("longwave"         ).first;
    const bool switch_fluxes            = command_line_options.at("fluxes"           ).first;
    const bool switch_cloud_optics      = command_line_options.at("cloud-optics"     ).first;
    const bool switch_output_optical    = command_line_options.at("output-optical"   ).first;
    const bool switch_output_bnd_fluxes = command_line_options.at("output-bnd-fluxes").first;

    // Print the options to the screen.
    print_command_line_options(command_line_options);


    ////// READ THE ATMOSPHERIC DATA //////
    Status::print_message("Reading atmospheric input data from NetCDF.");

    Netcdf_file input_nc("rte_rrtmgp_input.nc", Netcdf_mode::Read);

    const int n_x = input_nc.get_dimension_size("x");
    const int n_y = input_nc.get_dimension_size("y");
    const int n_lay_in = input_nc.get_dimension_size("lay");
    const int n_lev_in = input_nc.get_dimension_size("lev");
    const int n_col = n_x*n_y;

    Array<TF,1> xh;
    Array<TF,1> yh;
    Array<TF,1> zh;
    Array<TF,1> z;
    if (switch_tilting)
    {
        xh.set_dims({n_x+1});
        xh = std::move(input_nc.get_variable<TF>("xh", {n_x+1})); 
        yh.set_dims({n_y+1});
        yh = std::move(input_nc.get_variable<TF>("yh", {n_y+1})); 
        zh.set_dims({n_lev_in});
        zh = std::move(input_nc.get_variable<TF>("zh", {n_lev_in})); 
        z.set_dims({n_lay_in});
        z = std::move(input_nc.get_variable<TF>("z", {n_lay_in})); 
    }

	// Read the atmospheric fields.
    Array<TF,2> p_lay(input_nc.get_variable<TF>("p_lay", {n_lay_in, n_y, n_x}), {n_col, n_lay_in});
    Array<TF,2> t_lay(input_nc.get_variable<TF>("t_lay", {n_lay_in, n_y, n_x}), {n_col, n_lay_in});
    Array<TF,2> p_lev(input_nc.get_variable<TF>("p_lev", {n_lev_in, n_y, n_x}), {n_col, n_lev_in});
    Array<TF,2> t_lev(input_nc.get_variable<TF>("t_lev", {n_lev_in, n_y, n_x}), {n_col, n_lev_in});
    
	// Fetch the col_dry in case present.
    Array<TF,2> col_dry;
    if (input_nc.variable_exists("col_dry"))
    {
        col_dry.set_dims({n_col, n_lay_in});
        col_dry = std::move(input_nc.get_variable<TF>("col_dry", {n_lay_in, n_y, n_x}));
    }

    // Create container for the gas concentrations and read gases.
    Gas_concs<TF> gas_concs;
    read_and_set_vmr("h2o", n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("co2", n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("o3" , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("n2o", n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("co" , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("ch4", n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("o2" , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("n2" , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);

    read_and_set_vmr("ccl4"   , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("cfc11"  , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("cfc12"  , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("cfc22"  , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("hfc143a", n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("hfc125" , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("hfc23"  , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("hfc32"  , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("hfc134a", n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("cf4"    , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    read_and_set_vmr("no2"    , n_y, n_x, n_col, n_lay_in, input_nc, gas_concs);
    
    Array<TF,2> lwp;
    Array<TF,2> iwp;
    Array<TF,2> rel;
    Array<TF,2> rei;

    if (switch_cloud_optics)
    {
        lwp.set_dims({n_col, n_lay_in});
        lwp = std::move(input_nc.get_variable<TF>("lwp", {n_lay_in, n_y, n_x}));

        iwp.set_dims({n_col, n_lay_in});
        iwp = std::move(input_nc.get_variable<TF>("iwp", {n_lay_in, n_y, n_x}));

        rel.set_dims({n_col, n_lay_in});
        rel = std::move(input_nc.get_variable<TF>("rel", {n_lay_in, n_y, n_x}));

        rei.set_dims({n_col, n_lay_in});
        rei = std::move(input_nc.get_variable<TF>("rei", {n_lay_in, n_y, n_x}));
    }
	
    int n_lev_tilt;
    int n_lay_tilt;
    if (switch_tilting)
    {
        Array<ijk,1> path;
        Array<TF,1> zh_tilt;
        tilted_path(xh.v(),yh.v(),zh.v(),z.v(),.55,1.3,path.v(),zh_tilt.v());
        n_lev_tilt = zh_tilt.v().size();
        n_lay_tilt = n_lev_tilt - 1;
        path.set_dims({n_lay_tilt}); 
        zh_tilt.set_dims({n_lev_tilt}); 
        if (switch_cloud_optics)
        {
            for (int ilay=1; ilay<=n_lay_in; ++ilay)    
            {
                TF dz = zh({ilay+1}) - zh({ilay});
                for (int icol=1; icol<=n_col; ++icol)    
                {
                    lwp({icol, ilay}) /= dz;
                    iwp({icol, ilay}) /= dz;
                }
            }
            create_tilted_columns(n_x, n_y, n_lay_in, n_lev_in, zh_tilt.v(), path.v(), lwp.v());
            create_tilted_columns(n_x, n_y, n_lay_in, n_lev_in, zh_tilt.v(), path.v(), iwp.v());
            create_tilted_columns(n_x, n_y, n_lay_in, n_lev_in, zh_tilt.v(), path.v(), rel.v());
            create_tilted_columns(n_x, n_y, n_lay_in, n_lev_in, zh_tilt.v(), path.v(), rei.v());
            lwp.expand_dims({n_col, n_lay_tilt});
            iwp.expand_dims({n_col, n_lay_tilt});
            rel.expand_dims({n_col, n_lay_tilt});
            rei.expand_dims({n_col, n_lay_tilt});
            
            for (int ilay=1; ilay<=n_lay_tilt; ++ilay)    
            {
                TF dz = zh_tilt({ilay+1}) - zh_tilt({ilay});
                for (int icol=1; icol<=n_col; ++icol)    
                {
                    lwp({icol, ilay}) *= dz;
                    iwp({icol, ilay}) *= dz;
                }
            }
        }
        
        std::vector<std::string> gases = gas_concs.gas_names();
        for (int igas=0; igas<gases.size(); ++igas)
        {
            const Array<TF,2>& gas = gas_concs.get_vmr(gases[igas]);
            if (gas.size() == n_lay_in*n_col)
            {
                Array<TF,2> gas_tmp(gas);
                create_tilted_columns(n_x, n_y, n_lay_in, n_lev_in, zh_tilt.v(), path.v(), gas_tmp.v());
                gas_tmp.expand_dims({n_col, n_lay_tilt});
                gas_concs.remove_vmr(gases[igas]);
                gas_concs.set_vmr(gases[igas], gas_tmp);
            
            }
            else
            {
                throw std::runtime_error("oh no!!! I don't have a tilted column implementation for single profiles yet :'(");
            }
        }
        
        // create tilted columns of T and p. Important, create T first!!
        create_tilted_columns_levlay(n_x, n_y, n_lay_in, n_lev_in, zh.v(), z.v(), zh_tilt.v(), path.v(), t_lay.v(), t_lev.v());
        create_tilted_columns_levlay(n_x, n_y, n_lay_in, n_lev_in, zh.v(), z.v(), zh_tilt.v(), path.v(), p_lay.v(), p_lev.v());
        t_lay.expand_dims({n_col, n_lay_tilt});
        t_lev.expand_dims({n_col, n_lev_tilt});
        p_lay.expand_dims({n_col, n_lay_tilt});
        p_lev.expand_dims({n_col, n_lev_tilt});
    }
    
    const int n_lev = (switch_tilting) ? n_lev_tilt : n_lev_in;
    const int n_lay = (switch_tilting) ? n_lay_tilt : n_lay_in;

    ////// CREATE THE OUTPUT FILE //////
    // Create the general dimensions and arrays.

    Status::print_message("Preparing NetCDF output file.");

    Netcdf_file output_nc("rte_rrtmgp_output.nc", Netcdf_mode::Create);
    output_nc.add_dimension("y", n_y);
    output_nc.add_dimension("x", n_x);
    output_nc.add_dimension("lay", n_lay);
    output_nc.add_dimension("lev", n_lev);
    output_nc.add_dimension("pair", 2);

    std::cout<<"-----------------"<<std::endl;	
    auto nc_lay = output_nc.add_variable<TF>("p_lay", {"lay", "y", "x"});
    auto nc_lev = output_nc.add_variable<TF>("p_lev", {"lev", "y", "x"});

    nc_lay.insert(p_lay.v(), {0, 0, 0}, {n_lay, n_y, n_x});
    nc_lev.insert(p_lev.v(), {0, 0, 0}, {n_lev, n_y, n_x});
    std::cout<<"-----------------"<<std::endl;	
    ////// RUN THE LONGWAVE SOLVER //////
    if (switch_longwave)
    {
        // Initialize the solver.
        Status::print_message("Initializing the longwave solver.");
        Radiation_solver_longwave<TF> rad_lw(gas_concs, "coefficients_lw.nc", "cloud_coefficients_lw.nc");

        // Read the boundary conditions.
        const int n_bnd_lw = rad_lw.get_n_bnd();
        const int n_gpt_lw = rad_lw.get_n_gpt();

        Array<TF,2> emis_sfc(input_nc.get_variable<TF>("emis_sfc", {n_y, n_x, n_bnd_lw}), {n_bnd_lw, n_col});
        Array<TF,1> t_sfc(input_nc.get_variable<TF>("t_sfc", {n_y, n_x}), {n_col});

        // Create output arrays.
        Array<TF,3> lw_tau;
        Array<TF,3> lay_source;
        Array<TF,3> lev_source_inc;
        Array<TF,3> lev_source_dec;
        Array<TF,2> sfc_source;

        if (switch_output_optical)
        {
            lw_tau        .set_dims({n_col, n_lay, n_gpt_lw});
            lay_source    .set_dims({n_col, n_lay, n_gpt_lw});
            lev_source_inc.set_dims({n_col, n_lay, n_gpt_lw});
            lev_source_dec.set_dims({n_col, n_lay, n_gpt_lw});
            sfc_source    .set_dims({n_col, n_gpt_lw});
        }

        Array<TF,2> lw_flux_up;
        Array<TF,2> lw_flux_dn;
        Array<TF,2> lw_flux_net;

        if (switch_fluxes)
        {
            lw_flux_up .set_dims({n_col, n_lev});
            lw_flux_dn .set_dims({n_col, n_lev});
            lw_flux_net.set_dims({n_col, n_lev});
        }

        Array<TF,3> lw_bnd_flux_up;
        Array<TF,3> lw_bnd_flux_dn;
        Array<TF,3> lw_bnd_flux_net;

        if (switch_output_bnd_fluxes)
        {
            lw_bnd_flux_up .set_dims({n_col, n_lev, n_bnd_lw});
            lw_bnd_flux_dn .set_dims({n_col, n_lev, n_bnd_lw});
            lw_bnd_flux_net.set_dims({n_col, n_lev, n_bnd_lw});
        }


        // Solve the radiation.
        Status::print_message("Solving the longwave radiation.");

        auto time_start = std::chrono::high_resolution_clock::now();

        // solve each y-slice independently
	#pragma omp parallel for
	for (int ix = 1; ix <= n_x; ++ix)
	{
           rad_lw.solve(
                ix, n_y,
		switch_fluxes,
                switch_cloud_optics,
                switch_output_optical,
                switch_output_bnd_fluxes,
                gas_concs,
                p_lay, p_lev,
                t_lay, t_lev,
                col_dry,
                t_sfc, emis_sfc,
                lwp, iwp,
                rel, rei,
                lw_tau, lay_source, lev_source_inc, lev_source_dec, sfc_source,
                lw_flux_up, lw_flux_dn, lw_flux_net,
                lw_bnd_flux_up, lw_bnd_flux_dn, lw_bnd_flux_net);
	}

        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(time_end-time_start).count();

        Status::print_message("Duration longwave solver: " + std::to_string(duration) + " (ms)");


        // Store the output.
        Status::print_message("Storing the longwave output.");

        output_nc.add_dimension("gpt_lw", n_gpt_lw);
        output_nc.add_dimension("band_lw", n_bnd_lw);

        auto nc_lw_band_lims_wvn = output_nc.add_variable<TF>("lw_band_lims_wvn", {"band_lw", "pair"});
        nc_lw_band_lims_wvn.insert(rad_lw.get_band_lims_wavenumber().v(), {0, 0});

        if (switch_output_optical)
        {
            auto nc_lw_band_lims_gpt = output_nc.add_variable<int>("lw_band_lims_gpt", {"band_lw", "pair"});
            nc_lw_band_lims_gpt.insert(rad_lw.get_band_lims_gpoint().v(), {0, 0});

            auto nc_lw_tau = output_nc.add_variable<TF>("lw_tau", {"gpt_lw", "lay", "y", "x"});
            nc_lw_tau.insert(lw_tau.v(), {0, 0, 0, 0}, {n_gpt_lw, n_lay, n_y, n_x});

            auto nc_lay_source     = output_nc.add_variable<TF>("lay_source"    , {"gpt_lw", "lay", "y", "x"});
            auto nc_lev_source_inc = output_nc.add_variable<TF>("lev_source_inc", {"gpt_lw", "lay", "y", "x"});
            auto nc_lev_source_dec = output_nc.add_variable<TF>("lev_source_dec", {"gpt_lw", "lay", "y", "x"});

            auto nc_sfc_source = output_nc.add_variable<TF>("sfc_source", {"gpt_lw", "y", "x"});

            nc_lay_source.insert    (lay_source.v()    , {0, 0, 0, 0}, {n_gpt_lw, n_lay, n_y, n_x});
            nc_lev_source_inc.insert(lev_source_inc.v(), {0, 0, 0, 0}, {n_gpt_lw, n_lay, n_y, n_x});
            nc_lev_source_dec.insert(lev_source_dec.v(), {0, 0, 0, 0}, {n_gpt_lw, n_lay, n_y, n_x});

            nc_sfc_source.insert(sfc_source.v(), {0, 0, 0}, {n_gpt_lw, n_y, n_x});
        }

        if (switch_fluxes)
        {
            auto nc_lw_flux_up  = output_nc.add_variable<TF>("lw_flux_up" , {"lev", "y", "x"});
            auto nc_lw_flux_dn  = output_nc.add_variable<TF>("lw_flux_dn" , {"lev", "y", "x"});
            auto nc_lw_flux_net = output_nc.add_variable<TF>("lw_flux_net", {"lev", "y", "x"});

            nc_lw_flux_up .insert(lw_flux_up .v(), {0, 0, 0}, {n_lev, n_y, n_x});
            nc_lw_flux_dn .insert(lw_flux_dn .v(), {0, 0, 0}, {n_lev, n_y, n_x});
            nc_lw_flux_net.insert(lw_flux_net.v(), {0, 0, 0}, {n_lev, n_y, n_x});

            if (switch_output_bnd_fluxes)
            {
                auto nc_lw_bnd_flux_up  = output_nc.add_variable<TF>("lw_bnd_flux_up" , {"band_lw", "lev", "y", "x"});
                auto nc_lw_bnd_flux_dn  = output_nc.add_variable<TF>("lw_bnd_flux_dn" , {"band_lw", "lev", "y", "x"});
                auto nc_lw_bnd_flux_net = output_nc.add_variable<TF>("lw_bnd_flux_net", {"band_lw", "lev", "y", "x"});

                nc_lw_bnd_flux_up .insert(lw_bnd_flux_up .v(), {0, 0, 0, 0}, {n_bnd_lw, n_lev, n_y, n_x});
                nc_lw_bnd_flux_dn .insert(lw_bnd_flux_dn .v(), {0, 0, 0, 0}, {n_bnd_lw, n_lev, n_y, n_x});
                nc_lw_bnd_flux_net.insert(lw_bnd_flux_net.v(), {0, 0, 0, 0}, {n_bnd_lw, n_lev, n_y, n_x});
            }
        }
    }


    ////// RUN THE SHORTWAVE SOLVER //////
    if (switch_shortwave)
    {
        // Initialize the solver.
        Status::print_message("Initializing the shortwave solver.");

        Radiation_solver_shortwave<TF> rad_sw(gas_concs, "coefficients_sw.nc", "cloud_coefficients_sw.nc");

        // Read the boundary conditions.
        const int n_bnd_sw = rad_sw.get_n_bnd();
        const int n_gpt_sw = rad_sw.get_n_gpt();

        Array<TF,1> mu0(input_nc.get_variable<TF>("mu0", {n_y, n_x}), {n_col});
        Array<TF,2> sfc_alb_dir(input_nc.get_variable<TF>("sfc_alb_dir", {n_y, n_x, n_bnd_sw}), {n_bnd_sw, n_col});
        Array<TF,2> sfc_alb_dif(input_nc.get_variable<TF>("sfc_alb_dif", {n_y, n_x, n_bnd_sw}), {n_bnd_sw, n_col});

        Array<TF,1> tsi_scaling({n_col});
        if (input_nc.variable_exists("tsi"))
        {
            Array<TF,1> tsi(input_nc.get_variable<TF>("tsi", {n_y, n_x}), {n_col});
            const TF tsi_ref = rad_sw.get_tsi();
            for (int icol=1; icol<=n_col; ++icol)
                tsi_scaling({icol}) = tsi({icol}) / tsi_ref;
        }
        else
        {
            for (int icol=1; icol<=n_col; ++icol)
                tsi_scaling({icol}) = TF(1.);
        }

        // Create output arrays.
        Array<TF,3> sw_tau;
        Array<TF,3> ssa;
        Array<TF,3> g;
        Array<TF,2> toa_source;

        if (switch_output_optical)
        {
            sw_tau    .set_dims({n_col, n_lay, n_gpt_sw});
            ssa       .set_dims({n_col, n_lay, n_gpt_sw});
            g         .set_dims({n_col, n_lay, n_gpt_sw});
            toa_source.set_dims({n_col, n_gpt_sw});
        }

        Array<TF,2> sw_flux_up;
        Array<TF,2> sw_flux_dn;
        Array<TF,2> sw_flux_dn_dir;
        Array<TF,2> sw_flux_net;

        if (switch_fluxes)
        {
            sw_flux_up    .set_dims({n_col, n_lev});
            sw_flux_dn    .set_dims({n_col, n_lev});
            sw_flux_dn_dir.set_dims({n_col, n_lev});
            sw_flux_net   .set_dims({n_col, n_lev});
        }

        Array<TF,3> sw_bnd_flux_up;
        Array<TF,3> sw_bnd_flux_dn;
        Array<TF,3> sw_bnd_flux_dn_dir;
        Array<TF,3> sw_bnd_flux_net;

        if (switch_output_bnd_fluxes)
        {
            sw_bnd_flux_up    .set_dims({n_col, n_lev, n_bnd_sw});
            sw_bnd_flux_dn    .set_dims({n_col, n_lev, n_bnd_sw});
            sw_bnd_flux_dn_dir.set_dims({n_col, n_lev, n_bnd_sw});
            sw_bnd_flux_net   .set_dims({n_col, n_lev, n_bnd_sw});
        }


        // Solve the radiation.
        Status::print_message("Solving the shortwave radiation.");

        auto time_start = std::chrono::high_resolution_clock::now();
	
        #pragma omp parallel for
        for (int ix = 1; ix <= n_x; ++ix)
        {
            rad_sw.solve(
		ix, n_y,
                switch_fluxes,
                switch_cloud_optics,
                switch_output_optical,
                switch_output_bnd_fluxes,
                gas_concs,
                p_lay, p_lev,
                t_lay, t_lev,
                col_dry,
                sfc_alb_dir, sfc_alb_dif,
                tsi_scaling, mu0,
                lwp, iwp,
                rel, rei,
                sw_tau, ssa, g,
                toa_source,
                sw_flux_up, sw_flux_dn,
                sw_flux_dn_dir, sw_flux_net,
                sw_bnd_flux_up, sw_bnd_flux_dn,
                sw_bnd_flux_dn_dir, sw_bnd_flux_net);
	}

        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(time_end-time_start).count();

        Status::print_message("Duration shortwave solver: " + std::to_string(duration) + " (ms)");


        // Store the output.
        Status::print_message("Storing the shortwave output.");

        output_nc.add_dimension("gpt_sw", n_gpt_sw);
        output_nc.add_dimension("band_sw", n_bnd_sw);

        auto nc_sw_band_lims_wvn = output_nc.add_variable<TF>("sw_band_lims_wvn", {"band_sw", "pair"});
        nc_sw_band_lims_wvn.insert(rad_sw.get_band_lims_wavenumber().v(), {0, 0});
        if (switch_output_optical)
        {
            auto nc_sw_band_lims_gpt = output_nc.add_variable<int>("sw_band_lims_gpt", {"band_sw", "pair"});
            nc_sw_band_lims_gpt.insert(rad_sw.get_band_lims_gpoint().v(), {0, 0});

            auto nc_sw_tau = output_nc.add_variable<TF>("sw_tau", {"gpt_sw", "lay", "y", "x"});
            auto nc_ssa    = output_nc.add_variable<TF>("ssa"   , {"gpt_sw", "lay", "y", "x"});
            auto nc_g      = output_nc.add_variable<TF>("g"     , {"gpt_sw", "lay", "y", "x"});

            nc_sw_tau.insert(sw_tau.v(), {0, 0, 0, 0}, {n_gpt_sw, n_lay, n_y, n_x});
            nc_ssa   .insert(ssa   .v(), {0, 0, 0, 0}, {n_gpt_sw, n_lay, n_y, n_x});
            nc_g     .insert(g     .v(), {0, 0, 0, 0}, {n_gpt_sw, n_lay, n_y, n_x});

            auto nc_toa_source = output_nc.add_variable<TF>("toa_source", {"gpt_sw", "y", "x"});
            nc_toa_source.insert(toa_source.v(), {0, 0,0}, {n_gpt_sw, n_y, n_x});
        }

        if (switch_fluxes)
        {
            auto nc_sw_flux_up     = output_nc.add_variable<TF>("sw_flux_up"    , {"lev", "y", "x"});
            auto nc_sw_flux_dn     = output_nc.add_variable<TF>("sw_flux_dn"    , {"lev", "y", "x"});
            auto nc_sw_flux_dn_dir = output_nc.add_variable<TF>("sw_flux_dn_dir", {"lev", "y", "x"});
            auto nc_sw_flux_net    = output_nc.add_variable<TF>("sw_flux_net"   , {"lev", "y", "x"});
//			std::cout<<n_lev*n_y*n_x<<" - "<<sw_flux_up.v().size()<<std::endl;
            nc_sw_flux_up    .insert(sw_flux_up    .v(), {0, 0, 0}, {n_lev, n_y, n_x});
            nc_sw_flux_dn    .insert(sw_flux_dn    .v(), {0, 0, 0}, {n_lev, n_y, n_x});
            nc_sw_flux_dn_dir.insert(sw_flux_dn_dir.v(), {0, 0, 0}, {n_lev, n_y, n_x});
            nc_sw_flux_net   .insert(sw_flux_net   .v(), {0, 0, 0}, {n_lev, n_y, n_x});

            if (switch_output_bnd_fluxes)
            {
                auto nc_sw_bnd_flux_up     = output_nc.add_variable<TF>("sw_bnd_flux_up"    , {"band_sw", "lev", "y", "x"});
                auto nc_sw_bnd_flux_dn     = output_nc.add_variable<TF>("sw_bnd_flux_dn"    , {"band_sw", "lev", "y", "x"});
                auto nc_sw_bnd_flux_dn_dir = output_nc.add_variable<TF>("sw_bnd_flux_dn_dir", {"band_sw", "lev", "y", "x"});
                auto nc_sw_bnd_flux_net    = output_nc.add_variable<TF>("sw_bnd_flux_net"   , {"band_sw", "lev", "y", "x"});

                nc_sw_bnd_flux_up    .insert(sw_bnd_flux_up    .v(), {0, 0, 0, 0}, {n_bnd_sw, n_lev, n_y, n_x});
                nc_sw_bnd_flux_dn    .insert(sw_bnd_flux_dn    .v(), {0, 0, 0, 0}, {n_bnd_sw, n_lev, n_y, n_x});
                nc_sw_bnd_flux_dn_dir.insert(sw_bnd_flux_dn_dir.v(), {0, 0, 0, 0}, {n_bnd_sw, n_lev, n_y, n_x});
                nc_sw_bnd_flux_net   .insert(sw_bnd_flux_net   .v(), {0, 0, 0, 0}, {n_bnd_sw, n_lev, n_y, n_x});
            }
        }
    }

    Status::print_message("###### Finished RTE+RRTMGP solver ######");
}


int main(int argc, char** argv)
{
    try
    {
        solve_radiation<FLOAT_TYPE>(argc, argv);
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
