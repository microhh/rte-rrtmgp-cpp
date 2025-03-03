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
#include <cmath>
#include <iomanip>
#include <cuda_profiler_api.h>


#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "Gas_concs.h"
#include "tilted_column.h"
#include "types.h"

void read_and_set_vmr(
        const std::string& gas_name, const int n_col_x, const int n_col_y, const int n_lay,
        const Netcdf_handle& input_nc, Gas_concs& gas_concs)
{
    const std::string vmr_gas_name = "vmr_" + gas_name;

    if (input_nc.variable_exists(vmr_gas_name))
    {
        std::map<std::string, int> dims = input_nc.get_variable_dimensions(vmr_gas_name);
        const int n_dims = dims.size();

        if (n_dims == 0)
        {
            gas_concs.set_vmr(gas_name, input_nc.get_variable<Float>(vmr_gas_name));
        }
        else if (n_dims == 1)
        {
            if (dims.at("lay") == n_lay)
                gas_concs.set_vmr(gas_name,
                        Array<Float,1>(input_nc.get_variable<Float>(vmr_gas_name, {n_lay}), {n_lay}));
            else
                throw std::runtime_error("Illegal dimensions of gas \"" + gas_name + "\" in input");
        }
        else if (n_dims == 3)
        {
            if (dims.at("lay") == n_lay && dims.at("y") == n_col_y && dims.at("x") == n_col_x)
                gas_concs.set_vmr(gas_name,
                        Array<Float,2>(input_nc.get_variable<Float>(vmr_gas_name, {n_lay, n_col_y, n_col_x}), {n_col_x * n_col_y, n_lay}));
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
        std::map<std::string, std::pair<bool, std::string>>& command_line_switches,
        std::map<std::string, std::pair<int, std::string>>& command_line_ints,
        int argc, char** argv)
{
    for (int i=1; i<argc; ++i)
    {
        std::string argument(argv[i]);
        boost::trim(argument);

        if (argument == "-h" || argument == "--help")
        {
            Status::print_message("Possible usage:");
            for (const auto& clo : command_line_switches)
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

        if (command_line_switches.find(argument) == command_line_switches.end())
        {
            std::string error = argument + " is an illegal command line option.";
            throw std::runtime_error(error);
        }
        else
        {
            command_line_switches.at(argument).first = enable;
        }

        // Check if a is integer is too be expect and if so, supplied
        if (command_line_ints.find(argument) != command_line_ints.end() && i+1 < argc)
        {
            std::string next_argument(argv[i+1]);
            boost::trim(next_argument);

            bool arg_is_int = true;
            for (int j=0; j<next_argument.size(); ++j)
                arg_is_int *= std::isdigit(next_argument[j]);

            if (arg_is_int)
            {
                command_line_ints.at(argument).first = std::stoi(argv[i+1]);
                ++i;
            }
        }
    }

    return false;
}

void print_command_line_options(
        const std::map<std::string, std::pair<bool, std::string>>& command_line_switches,
        const std::map<std::string, std::pair<int, std::string>>& command_line_ints)
{
    Status::print_message("Solver settings:");
    for (const auto& option : command_line_switches)
    {
        std::ostringstream ss;
        ss << std::left << std::setw(20) << (option.first);
        if (command_line_ints.find(option.first) != command_line_ints.end() && option.second.first)
            ss << " = " << std::boolalpha << command_line_ints.at(option.first).first << std::endl;
        else
            ss << " = " << std::boolalpha << option.second.first << std::endl;
        Status::print_message(ss);
   }
}

std::vector<Float> linspace(Float start, Float end, int num_points) {
    std::vector<Float> result;
    if (num_points <= 0) {
        return result; // Return empty vector for invalid input
    }
    if (num_points == 1) {
        result.push_back(start);
        return result;
    }

    Float step = (end - start) / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}

void tilt_input(int argc, char** argv)
{
    Status::print_message("###### Starting Script ######");

    ////// FLOW CONTROL SWITCHES //////
    // Parse the command line options.
    std::map<std::string, std::pair<bool, std::string>> command_line_switches {
        {"cloud-optics"      , { false, "Enable cloud optics (both liquid and ice)."}},
        {"liq-cloud-optics"  , { false, "liquid only cloud optics."                 }},
        {"ice-cloud-optics"  , { false, "ice only cloud optics."                    }},
        {"multi-start-point" , { true, "average tilted grids for different tilt path start points within bottom cell."  }},
        {"interpolation"     , { true, "interpolate to constant dz grid."                    }},
        {"match-lay-size"      , { false, "keep input file size" }},
        {"tilt-sza"          , { false, "tilt provided value of sza in input file. IN DEGREES. '--tilt-sza 50': use a sza of 50 degrees" }},
        {"tilt-azi"          , { false, "tilt provided value of azi in input file. FROM POS Y, CLOCKWISE, IN DEGREES. '--tilt-azi 240': use of azi of 240 degrees"   }}};

    std::map<std::string, std::pair<int, std::string>> command_line_ints {
        {"tilt-sza", {0, "sza in degrees."}},
        {"tilt-azi", {0 , "azi in degrees" }}};

    if (parse_command_line_options(command_line_switches, command_line_ints, argc, argv))
        return;

    bool switch_cloud_optics      = command_line_switches.at("cloud-optics"      ).first;
    bool switch_liq_cloud_optics  = command_line_switches.at("liq-cloud-optics"  ).first;
    bool switch_ice_cloud_optics  = command_line_switches.at("ice-cloud-optics"  ).first;
    const bool switch_multi = command_line_switches.at("multi-start-point").first;
    const bool switch_interpolation = command_line_switches.at("interpolation").first;
    const bool switch_match_lay_size = command_line_switches.at("match-lay-size").first;
    const bool tilt_sza             = command_line_switches.at("tilt-sza"    ).first;
    const bool tilt_azi             = command_line_switches.at("tilt-azi"    ).first;

    if (switch_cloud_optics)
    {
        switch_liq_cloud_optics = true;
        switch_ice_cloud_optics = true;
    }
    if (switch_liq_cloud_optics || switch_ice_cloud_optics)
    {
        switch_cloud_optics = true;
    }
    if (tilt_sza && !tilt_azi) {
        std::string error = "If tilt-sza is provided, user must pass tilt-azi too.";
        throw std::runtime_error(error);
    }

    // Print the options to the screen.
    print_command_line_options(command_line_switches, command_line_ints);

    Float sza = 0;
    Float azi = 0;

    if (tilt_sza) 
    {
        int sza_deg = Int(command_line_ints.at("tilt-sza").first);
        sza = sza_deg * 3.14159f / 180.0f;
    }
    if (tilt_azi) 
    {
        int azi_deg = Int(command_line_ints.at("tilt-azi").first);
        azi = azi_deg * 3.14159f / 180.0f;
    }

    ////// READ THE ATMOSPHERIC DATA //////
    Status::print_message("Reading atmospheric input data from NetCDF.");

    Netcdf_file input_nc("rte_rrtmgp_input.nc", Netcdf_mode::Read);
    const int n_col_x = input_nc.get_dimension_size("x");
    const int n_col_y = input_nc.get_dimension_size("y");
    const int n_col = n_col_x * n_col_y;
    const int n_lay = input_nc.get_dimension_size("lay");
    const int n_lev = input_nc.get_dimension_size("lev");
    const int n_z_in = input_nc.get_dimension_size("z");
    const int n_zh_in = input_nc.get_dimension_size("zh");


    Array<Float,1> xh;
    Array<Float,1> yh;
    Array<Float,1> zh;
    Array<Float,1> z;

    xh.set_dims({n_col_x+1});
    xh = std::move(input_nc.get_variable<Float>("xh", {n_col_x+1})); 
    yh.set_dims({n_col_y+1});
    yh = std::move(input_nc.get_variable<Float>("yh", {n_col_y+1})); 

    zh.set_dims({n_zh_in});
    zh = std::move(input_nc.get_variable<Float>("zh", {n_zh_in})); 
    z.set_dims({n_z_in});
    z = std::move(input_nc.get_variable<Float>("z", {n_z_in})); 

    // Read the atmospheric fields.
    Array<Float,2> p_lay(input_nc.get_variable<Float>("p_lay", {n_lay, n_col_y, n_col_x}), {n_col, n_lay});
    Array<Float,2> t_lay(input_nc.get_variable<Float>("t_lay", {n_lay, n_col_y, n_col_x}), {n_col, n_lay});
    Array<Float,2> p_lev(input_nc.get_variable<Float>("p_lev", {n_lev, n_col_y, n_col_x}), {n_col, n_lev});
    Array<Float,2> t_lev(input_nc.get_variable<Float>("t_lev", {n_lev, n_col_y, n_col_x}), {n_col, n_lev});

    // Fetch the col_dry in case present.
    Array<Float,2> col_dry;
    if (input_nc.variable_exists("col_dry"))
    {
        col_dry.set_dims({n_col, n_lay});
        col_dry = std::move(input_nc.get_variable<Float>("col_dry", {n_lay, n_col_y, n_col_x}));
    }

    // Create container for the gas concentrations and read gases.
    Gas_concs gas_concs;

    read_and_set_vmr("h2o", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("co2", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("o3" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("n2o", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("co" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("ch4", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("o2" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("n2" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);

    read_and_set_vmr("ccl4"   , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc11"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc12"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc22"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc143a", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc125" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc23"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc32"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc134a", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cf4"    , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("no2"    , n_col_x, n_col_y, n_lay, input_nc, gas_concs);

    Array<Float,2> lwp;
    Array<Float,2> iwp;
    Array<Float,2> rel;
    Array<Float,2> dei;

    if (switch_cloud_optics)
    {

        if (switch_liq_cloud_optics)
        {
            lwp.set_dims({n_col, n_lay});
            lwp = std::move(input_nc.get_variable<Float>("lwp", {n_lay, n_col_y, n_col_x}));

            rel.set_dims({n_col, n_lay});
            rel = std::move(input_nc.get_variable<Float>("rel", {n_lay, n_col_y, n_col_x}));
        }

        if (switch_ice_cloud_optics)
        {
            iwp.set_dims({n_col, n_lay});
            iwp = std::move(input_nc.get_variable<Float>("iwp", {n_lay, n_col_y, n_col_x}));

            dei.set_dims({n_col, n_lay});
            dei = std::move(input_nc.get_variable<Float>("dei", {n_lay, n_col_y, n_col_x}));
        }
    }

    std::array<std::pair<Float, Float>, 5> x_y_start_arr;
    x_y_start_arr[0] = std::make_pair(.5, .5);
    x_y_start_arr[1] = std::make_pair(0.01, 0.01);
    x_y_start_arr[2] = std::make_pair(0.01, 0.99);
    x_y_start_arr[3] = std::make_pair(0.99, 0.99);
    x_y_start_arr[4] = std::make_pair(0.99, 0.01);

    Float x_start;
    Float y_start;

    std::vector<std::string> gas_names = {
        "h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2", "ccl4", "cfc11", 
        "cfc12", "cfc22", "hfc143a", "hfc125", "hfc23", "hfc32", "hfc134a", 
        "cf4", "no2"
    };

    // this is a choice, user can pass a value or keep n_z_in and n_zh_in ?
    int n_lay_out;
    int n_lev_out;
    std::vector<Float> z_out;
    std::vector<Float> zh_out;
    if (switch_match_lay_size) {
        n_lay_out = n_z_in;
        n_lev_out = n_zh_in;
        z_out = linspace(z.v()[0], z.v()[n_z_in - 1], n_lay_out);
        zh_out = linspace(zh.v()[0], zh.v()[n_zh_in - 1], n_lev_out);
    }

    // Initialize containers for averaging
    std::vector<Array<Float,2>> rel_list, dei_list, lwp_list, iwp_list, t_lay_list, t_lev_list, p_lay_list, p_lev_list;
    std::map<std::string, std::vector<Array<Float,2>>> gas_concs_list;

    int n_lev_tilt;
    int n_lay_tilt;

    int loop_index = 0;
    for (const auto& pair : x_y_start_arr) {
        std::cout << "loop_index: " << loop_index << std::endl;
        Array<ijk,1> path;
        Array<Float,1> zh_tilt;

        Array<Float,2> lwp_copy = lwp;
        Array<Float,2> iwp_copy = iwp;
        Array<Float,2> rel_copy = rel;
        Array<Float,2> dei_copy = dei;
        Array<Float,2> t_lay_copy = t_lay;
        Array<Float,2> t_lev_copy = t_lev;
        Array<Float,2> p_lay_copy = p_lay;
        Array<Float,2> p_lev_copy = p_lev;
        Gas_concs gas_concs_copy = gas_concs;

        if (sza == 0 && loop_index > 0) {
            break;
        }
        if (!switch_multi && loop_index > 0)
        {
            break;
        }
        loop_index++;
        x_start = pair.first;
        y_start = pair.second;

        std::cout << "x_start: " << x_start << std::endl;
        std::cout << "y_start: " << y_start << std::endl;

        Status::print_message("###### Starting Tilting ######");
        auto time_start = std::chrono::high_resolution_clock::now();
        
        tilted_path(xh.v(),yh.v(),zh.v(),z.v(),sza,azi,x_start, y_start, path.v(),zh_tilt.v());
        std::cout << "finish tilted path" << std::endl;

        if (!switch_match_lay_size && loop_index == 1) {  
            n_lev_out = zh_tilt.v().size();
            n_lay_out = n_lev_out - 1;
            z_out = linspace(z.v()[0], z.v()[n_z_in - 1], n_lay_out);
            zh_out = linspace(zh.v()[0], zh.v()[n_zh_in - 1], n_lev_out);
        }

        n_lev_tilt = zh_tilt.v().size();
        n_lay_tilt = n_lev_tilt - 1;
        
        path.set_dims({n_lay_tilt}); 
        zh_tilt.set_dims({n_lev_tilt}); 
        if (switch_cloud_optics)
        {
            for (int ilay=1; ilay<=n_lay; ++ilay)    
            {
                Float dz = zh({ilay+1}) - zh({ilay});
                for (int icol=1; icol<=n_col; ++icol)    
                {
                    if (switch_liq_cloud_optics)
                    {
                        lwp_copy({icol, ilay}) /= dz;
                    }
                    if (switch_ice_cloud_optics)
                    {
                        iwp_copy({icol, ilay}) /= dz;
                    }
                }
            }
            if (switch_liq_cloud_optics)
            {
                create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), lwp_copy.v());
                create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), rel_copy.v());
                lwp_copy.expand_dims({n_col, n_lay_tilt});
                rel_copy.expand_dims({n_col, n_lay_tilt});
            }
            if (switch_ice_cloud_optics)
            {
                create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), iwp_copy.v());
                create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), dei_copy.v());

                iwp_copy.expand_dims({n_col, n_lay_tilt});
                dei_copy.expand_dims({n_col, n_lay_tilt});
            }

            for (int ilay=1; ilay<=n_lay_tilt; ++ilay)    
            {
                Float dz = zh_tilt({ilay+1}) - zh_tilt({ilay});
                for (int icol=1; icol<=n_col; ++icol)    
                {
                    if (switch_liq_cloud_optics)
                    {
                        lwp_copy({icol, ilay}) *= dz;
                    }
                    if (switch_ice_cloud_optics)
                    {
                        iwp_copy({icol, ilay}) *= dz;
                    }
                }
            }
        }

        for (const auto& gas_name : gas_names) {
            if (!gas_concs_copy.exists(gas_name)) {
                continue;
            }
            const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name); 

            if (gas.size() == n_lay*n_col) {
                Array<Float,2> gas_tmp(gas);
                create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), gas_tmp.v());
                gas_tmp.expand_dims({n_col, n_lay_tilt});
                gas_concs_copy.set_vmr(gas_name, gas_tmp);
            } 
            else if (gas.size() == 1) {
                // Do nothing for single profiles
            } 
            else {
                throw std::runtime_error("No tilted column implementation for single profiles.");
            }
        }

        // create tilted columns of T and p. Important, create T first!!
        // if t lev all 0, interpolate from t lay
        if (*std::max_element(t_lev_copy.v().begin(), t_lev_copy.v().end()) <= 0) {
            for (int i = 0; i < n_col; ++i) {
                for (int j = 1; j < n_lay; ++j) {
                    t_lev_copy({i, j}) = (t_lay_copy({i, j}) + t_lay_copy({i, j - 1})) / 2.0;
                }
                t_lev_copy({i, n_lev - 1}) = t_lay_copy({i, n_lay - 1});        
            }
        }
        // finish copies
        create_tilted_columns_levlay(n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), t_lay_copy.v(), t_lev_copy.v());
        create_tilted_columns_levlay(n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), p_lay_copy.v(), p_lev_copy.v());
        t_lay_copy.expand_dims({n_col, n_lay_tilt});
        t_lev_copy.expand_dims({n_col, n_lev_tilt});
        p_lay_copy.expand_dims({n_col, n_lay_tilt});
        p_lev_copy.expand_dims({n_col, n_lev_tilt});

        std::vector<Float> midpoints(n_lay_tilt);
        std::vector<Float> z_tilt(n_lay_tilt);
        for (int i = 1; i < n_lev_tilt; ++i) {
            midpoints[i - 1] = (zh_tilt.v()[i] - zh_tilt.v()[i - 1]) / 2.0;
        }
        for (int i = 0; i < n_lay_tilt; ++i) {
            z_tilt[i] = zh_tilt.v()[i] + midpoints[i];
        }

        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(time_end-time_start).count();

        Status::print_message("Duration tilting columns: " + std::to_string(duration) + " (ms)");
        if (switch_interpolation) {
            ////// INTERPOLATE TO CONSTANT DZ GRID //////
            Status::print_message("Start z grid regularization.");
    
            // Intrinsic Variables //
            interpolate_3D_field(n_col_x, n_col_y, z_tilt, z_out, t_lay_copy.v());
            t_lay_copy.expand_dims({n_col, n_lay_out});

            interpolate_3D_field(n_col_x, n_col_y, z_tilt, z_out, p_lay_copy.v());    
            p_lay_copy.expand_dims({n_col, n_lay_out});

            interpolate_3D_field(n_col_x, n_col_y, zh_tilt.v(), zh_out, t_lev_copy.v());
            t_lev_copy.expand_dims({n_col, n_lev_out});

            interpolate_3D_field(n_col_x, n_col_y, zh_tilt.v(), zh_out, p_lev_copy.v());
            p_lev_copy.expand_dims({n_col, n_lev_out});


            for (const auto& gas_name : gas_names) {
                if (!gas_concs_copy.exists(gas_name)) {
                    continue;
                }
                const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);
                std::string var_name = "vmr_" + gas_name;
                if (gas.size() > 1) {
                    Array<Float,2> gas_tmp(gas);
                    interpolate_3D_field(n_col_x, n_col_y, z_tilt, z_out, gas_tmp.v());
                    gas_tmp.expand_dims({n_col, n_lay_out});
                    gas_concs_copy.set_vmr(gas_name, gas_tmp);
                }
            }    
            
            if (switch_liq_cloud_optics)
            {
                interpolate_3D_field(n_col_x, n_col_y, z_tilt, z_out, rel_copy.v());
                rel_copy.expand_dims({n_col, n_lay_out});
            }
            if (switch_ice_cloud_optics)
            {
                interpolate_3D_field(n_col_x, n_col_y, z_tilt, z_out, dei_copy.v());
                dei_copy.expand_dims({n_col, n_lay_out});
            }


            // Extrinsic Variables //

            if (switch_cloud_optics)
            {
                for (int ilay=1; ilay<=n_lay_tilt; ++ilay)    
                {
                    Float dz = zh_tilt({ilay+1}) - zh_tilt({ilay});
                    for (int icol=1; icol<=n_col; ++icol)    
                    {
                        if (switch_liq_cloud_optics)
                        {
                            lwp_copy({icol, ilay}) /= dz;
                        }
                        if (switch_ice_cloud_optics)
                        {
                            iwp_copy({icol, ilay}) /= dz;
                        }
                    }
                }
                if (switch_liq_cloud_optics)
                {
                    interpolate_3D_field(n_col_x, n_col_y, z_tilt, z_out, lwp_copy.v());
                    lwp_copy.expand_dims({n_col, n_lay_out});
                }
                if (switch_ice_cloud_optics)
                {
                    interpolate_3D_field(n_col_x, n_col_y, z_tilt, z_out, iwp_copy.v());
                    iwp_copy.expand_dims({n_col, n_lay_out});
                }

                const Float dz_out = zh_out[1] - zh_out[0];
                
                for (int ilay=1; ilay<=n_lay_out; ++ilay)    
                {
                    for (int icol=1; icol<=n_col; ++icol)    
                    {
                        if (switch_liq_cloud_optics)
                        {
                            lwp_copy({icol, ilay}) *= dz_out;
                        }
                        if (switch_ice_cloud_optics)
                        {
                            iwp_copy({icol, ilay}) *= dz_out;
                        }
                    }
                }
            }
            Status::print_message("Finish z grid regularization.");
        }
    
        // append lwp, iwp, gases, t_lay, t_lev, p_lay, p_lev to list of arrays for averaging after we complete the loops
        if (switch_liq_cloud_optics) {
            lwp_list.push_back(lwp_copy);
            rel_list.push_back(rel_copy);
        }
        if (switch_ice_cloud_optics) {
            iwp_list.push_back(iwp_copy);
            dei_list.push_back(dei_copy);
        }
        t_lay_list.push_back(t_lay_copy);
        t_lev_list.push_back(t_lev_copy);
        p_lay_list.push_back(p_lay_copy);
        p_lev_list.push_back(p_lev_copy);

        for (const auto& gas_name : gas_names) {
            if (gas_concs_copy.exists(gas_name)) {
                const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);
                if (gas.size() > 1) 
                {
                    gas_concs_list[gas_name].push_back(gas_concs_copy.get_vmr(gas_name));
                }
            }
        }
        Status::print_message("Finish appending.");
    }

    Array<Float,2> lwp_avg({n_col, n_lay_out});
    Array<Float,2> rel_avg({n_col, n_lay_out});
    if (switch_liq_cloud_optics) {
        
        lwp_avg.fill(0.0);
        average_3D_field_list(n_col, n_lay_out, lwp_list, &lwp_avg);

        rel_avg.fill(0.0);
        average_3D_field_list(n_col, n_lay_out, rel_list, &rel_avg);
        Status::print_message("finish rel.");
        
    }
    Array<Float,2> iwp_avg({n_col, n_lay_out});
    Array<Float,2> dei_avg({n_col, n_lay_out});
    if (switch_ice_cloud_optics) {
        
        iwp_avg.fill(0.0);
        average_3D_field_list(n_col, n_lay_out, iwp_list, &iwp_avg);
        dei_avg.fill(0.0);
        average_3D_field_list(n_col, n_lay_out, dei_list, &dei_avg);
    }

    Array<Float,2> t_lay_avg({n_col, n_lay_out});
    t_lay_avg.fill(0.0);
    average_3D_field_list(n_col, n_lay_out, t_lay_list, &t_lay_avg);

    Array<Float,2> t_lev_avg({n_col, n_lev_out});
    t_lev_avg.fill(0.0);
    average_3D_field_list(n_col, n_lev_out, t_lev_list, &t_lev_avg);

    Array<Float,2> p_lay_avg({n_col, n_lay_out});
    p_lay_avg.fill(0.0);
    average_3D_field_list(n_col, n_lay_out, p_lay_list, &p_lay_avg);

    Array<Float,2> p_lev_avg({n_col, n_lev_out});
    p_lev_avg.fill(0.0);
    average_3D_field_list(n_col, n_lev_out, p_lev_list, &p_lev_avg);

    Status::print_message("Average the gases.");
    for (const auto& gas_name : gas_names) {
        if (gas_concs.exists(gas_name)) {
            const Array<Float,2>& gas = gas_concs.get_vmr(gas_name);
            if (gas.size() > 1) 
            {
                Array<Float,2> gas_avg({n_col, n_lay_out});
                gas_avg.fill(0.0);
                average_3D_field_list(n_col, n_lay_out, gas_concs_list[gas_name], &gas_avg);
                gas_concs.set_vmr(gas_name, gas_avg);
            }
        }
    }

    ////// CREATE THE OUTPUT FILE //////
    // Create the general dimensions and arrays.

    Status::print_message("Preparing NetCDF output file.");

    // Create the new NetCDF file
    Netcdf_file output_nc("rte_rrtmgp_tilted_input.nc", Netcdf_mode::Create);

    // Copy dimensions from the input file  
    const int n_bnd_sw = 14;
    const int n_bnd_lw = 16;
    output_nc.add_dimension("band_sw", n_bnd_sw);
    output_nc.add_dimension("band_lw", n_bnd_lw);

    output_nc.add_dimension("lay", n_lay_out);
    output_nc.add_dimension("lev", n_lev_out);

    output_nc.add_dimension("x", n_col_x);
    output_nc.add_dimension("y", n_col_y);
    output_nc.add_dimension("z", n_lay_out);

    output_nc.add_dimension("xh", n_col_x+1);
    output_nc.add_dimension("yh", n_col_y+1);
    output_nc.add_dimension("zh", n_lev_out);

    Array<Float,1> grid_x(input_nc.get_variable<Float>("x", {n_col_x}), {n_col_x});
    Array<Float,1> grid_xh(input_nc.get_variable<Float>("xh", {n_col_x+1}), {n_col_x+1});
    Array<Float,1> grid_y(input_nc.get_variable<Float>("y", {n_col_y}), {n_col_y});
    Array<Float,1> grid_yh(input_nc.get_variable<Float>("yh", {n_col_y+1}), {n_col_y+1});

    // Create and write the grid coordinate variables
    auto nc_x = output_nc.add_variable<Float>("x", {"x"});
    auto nc_y = output_nc.add_variable<Float>("y", {"y"});
    auto nc_xh = output_nc.add_variable<Float>("xh", {"xh"});
    auto nc_yh = output_nc.add_variable<Float>("yh", {"yh"});

    nc_x.insert(grid_x.v(), {0});
    nc_y.insert(grid_y.v(), {0});
    nc_xh.insert(grid_xh.v(), {0});
    nc_yh.insert(grid_yh.v(), {0});

    // Create and write surface properties
    Array<Float,2> sfc_alb_dir(input_nc.get_variable<Float>("sfc_alb_dir", {n_col_y, n_col_x, n_bnd_sw}), {n_bnd_sw, n_col});
    Array<Float,2> sfc_alb_dif(input_nc.get_variable<Float>("sfc_alb_dif", {n_col_y, n_col_x, n_bnd_sw}), {n_bnd_sw, n_col});
    Array<Float,2> emis_sfc(input_nc.get_variable<Float>("emis_sfc", {n_col_y, n_col_x, n_bnd_lw}), {n_bnd_lw, n_col});
    Array<Float,1> t_sfc(input_nc.get_variable<Float>("t_sfc", {n_col_y, n_col_x}), {n_col});

    auto nc_sfc_alb_dir = output_nc.add_variable<Float>("sfc_alb_dir", {"y", "x", "band_sw"});
    auto nc_sfc_alb_dif = output_nc.add_variable<Float>("sfc_alb_dif", {"y", "x", "band_sw"});
    auto nc_emis_sfc = output_nc.add_variable<Float>("emis_sfc", {"y", "x", "band_lw"});
    auto nc_t_sfc = output_nc.add_variable<Float>("t_sfc", {"y", "x"});

    nc_sfc_alb_dir.insert(sfc_alb_dir.v(), {0, 0, 0});
    nc_sfc_alb_dif.insert(sfc_alb_dif.v(), {0, 0, 0});
    nc_emis_sfc.insert(emis_sfc.v(), {0, 0, 0});
    nc_t_sfc.insert(t_sfc.v(), {0, 0});
    
    // Write tsi scaling
    auto nc_tsi_scaling = output_nc.add_variable<Float>("tsi_scaling", {});
    nc_tsi_scaling.insert(std::cos(sza), {0});

    // Create and write the variables for the tilted data
    auto nc_z = output_nc.add_variable<Float>("z", {"z"});
    auto nc_zh = output_nc.add_variable<Float>("zh", {"zh"});
    nc_zh.insert(zh_out, {0});
    nc_z.insert(z_out, {0}); 

    // Write the atmospheric fields
    auto nc_play = output_nc.add_variable<Float>("p_lay", {"lay", "y", "x"});
    auto nc_plev = output_nc.add_variable<Float>("p_lev", {"lev", "y", "x"});

    nc_play.insert(p_lay_avg.v(), {0, 0, 0});
    nc_plev.insert(p_lev_avg.v(), {0, 0, 0});

    auto nc_tlay = output_nc.add_variable<Float>("t_lay", {"lay", "y", "x"});
    auto nc_tlev = output_nc.add_variable<Float>("t_lev", {"lev", "y", "x"});

    nc_tlay.insert(t_lay_avg.v(), {0, 0, 0});
    nc_tlev.insert(t_lev_avg.v(), {0, 0, 0});

    // Write the cloud optical properties if applicable
    if (switch_cloud_optics) {
        if (switch_liq_cloud_optics) 
        {
            std::cout << "Add LWP" << std::endl;
            auto nc_lwp = output_nc.add_variable<Float>("lwp", {"lay", "y", "x"});
            nc_lwp.insert(lwp_avg.v(), {0, 0, 0});
            std::cout << "Add REL" << std::endl;
            auto nc_rel = output_nc.add_variable<Float>("rel", {"lay", "y", "x"});
            nc_rel.insert(rel_avg.v(), {0, 0, 0});
        }

        if (switch_ice_cloud_optics) 
        {
            auto nc_iwp = output_nc.add_variable<Float>("iwp", {"lay", "y", "x"});
            auto nc_dei = output_nc.add_variable<Float>("dei", {"lay", "y", "x"});
            nc_iwp.insert(iwp_avg.v(), {0, 0, 0});
            nc_dei.insert(dei_avg.v(), {0, 0, 0});
        }
    }

    // Write the gas concentrations
    for (const auto& gas_name : gas_names) {
        if (!gas_concs.exists(gas_name)) {
            continue;
        }
        const Array<Float,2>& gas = gas_concs.get_vmr(gas_name);
        std::string var_name = "vmr_" + gas_name;
        if (gas.size() == 1) {
            std::cout << "Adding scalar variable for " << gas_name << std::endl;
            auto nc_gas = output_nc.add_variable<Float>(var_name);
            nc_gas.insert(gas.v(), {});
        } else {
            std::cout << "Adding array variable for " << gas_name << std::endl;
            auto nc_gas = output_nc.add_variable<Float>(var_name, {"lay", "y", "x"});
            const std::vector<Float>& flat_data = gas.v();
            nc_gas.insert(flat_data, {0, 0, 0});
        }
    }
    auto nc_mu = output_nc.add_variable<Float>("mu0", {"y", "x"});
    auto nc_azi = output_nc.add_variable<Float>("azi", {"y", "x"});

    Array<Float, 2> mu_array(std::array<int, 2>{n_col_y, n_col_x});
    Array<Float, 2> azi_array(std::array<int, 2>{n_col_y, n_col_x});

    mu_array.fill(1.0);
    azi_array.fill(0.0);

    nc_mu.insert(mu_array.v(), {0, 0});
    nc_azi.insert(azi_array.v(), {0, 0});

    auto nc_ng_x = output_nc.add_variable<Float>("ngrid_x", {});
    auto nc_ng_y = output_nc.add_variable<Float>("ngrid_y", {});
    auto nc_ng_z = output_nc.add_variable<Float>("ngrid_z", {});

    nc_ng_x.insert(48, {0});
    nc_ng_y.insert(48, {0});
    nc_ng_z.insert(32, {0});


    output_nc.sync();

    Status::print_message("###### Finished tilting ######");
}



int main(int argc, char** argv)
{
    try
    {
        tilt_input(argc, argv);
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

