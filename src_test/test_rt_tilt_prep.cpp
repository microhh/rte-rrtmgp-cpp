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

bool prepare_netcdf(Netcdf_handle& input_nc, std::string file_name, int n_lay, int n_lev, int n_col_x, int n_col_y,
                    int n_zh, int n_z, 
                    Float sza, std::vector<Float> zh, std::vector<Float> z,
                    Array<Float,2> p_lay, Array<Float,2> t_lay, Array<Float,2> p_lev, Array<Float,2> t_lev, 
                    Array<Float,2> lwp, Array<Float,2> iwp, Array<Float,2> rel, Array<Float,2> dei, 
                    Gas_concs& gas_concs, std::vector<std::string> gas_names,
                    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics) {
    const int n_col = n_col_x * n_col_y;

    // Create the new NetCDF file
    Netcdf_file output_nc(file_name, Netcdf_mode::Create);
    // Copy dimensions from the input file  
    const int n_bnd_sw = 14;
    const int n_bnd_lw = 16;
    output_nc.add_dimension("band_sw", n_bnd_sw);
    output_nc.add_dimension("band_lw", n_bnd_lw);

    output_nc.add_dimension("lay", n_lay);
    output_nc.add_dimension("lev", n_lev);

    output_nc.add_dimension("x", n_col_x);
    output_nc.add_dimension("y", n_col_y);
    output_nc.add_dimension("z", n_z);

    output_nc.add_dimension("xh", n_col_x+1);
    output_nc.add_dimension("yh", n_col_y+1);
    output_nc.add_dimension("zh", n_zh);

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

    // Write the grid coordinates
    auto nc_z = output_nc.add_variable<Float>("z", {"z"});
    auto nc_zh = output_nc.add_variable<Float>("zh", {"zh"});
    nc_zh.insert(zh, {0});
    nc_z.insert(z, {0}); 

    // Write the atmospheric fields
    auto nc_play = output_nc.add_variable<Float>("p_lay", {"lay", "y", "x"});
    auto nc_plev = output_nc.add_variable<Float>("p_lev", {"lev", "y", "x"});

    nc_play.insert(p_lay.v(), {0, 0, 0});
    nc_plev.insert(p_lev.v(), {0, 0, 0});

    auto nc_tlay = output_nc.add_variable<Float>("t_lay", {"lay", "y", "x"});
    auto nc_tlev = output_nc.add_variable<Float>("t_lev", {"lev", "y", "x"});

    nc_tlay.insert(t_lay.v(), {0, 0, 0});
    nc_tlev.insert(t_lev.v(), {0, 0, 0});

    // Write the cloud optical properties if applicable
    if (switch_cloud_optics) {
        if (switch_liq_cloud_optics) 
        {
            auto nc_lwp = output_nc.add_variable<Float>("lwp", {"lay", "y", "x"});
            nc_lwp.insert(lwp.v(), {0, 0, 0});
            auto nc_rel = output_nc.add_variable<Float>("rel", {"lay", "y", "x"});
            nc_rel.insert(rel.v(), {0, 0, 0});
        }

        if (switch_ice_cloud_optics) 
        {
            auto nc_iwp = output_nc.add_variable<Float>("iwp", {"lay", "y", "x"});
            auto nc_dei = output_nc.add_variable<Float>("dei", {"lay", "y", "x"});
            nc_iwp.insert(iwp.v(), {0, 0, 0});
            nc_dei.insert(dei.v(), {0, 0, 0});
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
            auto nc_gas = output_nc.add_variable<Float>(var_name);
            nc_gas.insert(gas.v(), {}); 
        } else {
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
    return true;
}

void tilt_input(int argc, char** argv)
{
    std::vector<std::string> gas_names = {
        "h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2", "ccl4", "cfc11", 
        "cfc12", "cfc22", "hfc143a", "hfc125", "hfc23", "hfc32", "hfc134a", 
        "cf4", "no2"
    };

    Status::print_message("###### Starting Script ######");

    ////// FLOW CONTROL SWITCHES //////
    // Parse the command line options.
    std::map<std::string, std::pair<bool, std::string>> command_line_switches {
        {"cloud-optics"      , { false, "Enable cloud optics (both liquid and ice)."}},
        {"liq-cloud-optics"  , { false, "liquid only cloud optics."                 }},
        {"ice-cloud-optics"  , { false, "ice only cloud optics."                    }},
        {"multi-start-points", { true,  "int to determine number of start points, total points used = i**2. Default value is 1."  }},
        {"compress"          , { true,  "compress upper lays to keep number of z and zh the same" }},
        {"compress-bkg"      , { true,  "compress background profile" }},
        {"tilt-sza"          , { false, "tilt provided value of sza in input file. IN DEGREES. '--tilt-sza 50': use a sza of 50 degrees" }},
        {"tilt-azi"          , { false, "tilt provided value of azi in input file. FROM POS Y, CLOCKWISE, IN DEGREES. '--tilt-azi 240': use of azi of 240 degrees"   }}};

    std::map<std::string, std::pair<int, std::string>> command_line_ints {
        {"multi-start-points", {1, "i**2 start points."}},
        {"tilt-sza", {0, "sza in degrees."}},
        {"tilt-azi", {0 , "azi in degrees" }}};

    if (parse_command_line_options(command_line_switches, command_line_ints, argc, argv))
        return;

    bool switch_cloud_optics      = command_line_switches.at("cloud-optics"      ).first;
    bool switch_liq_cloud_optics  = command_line_switches.at("liq-cloud-optics"  ).first;
    bool switch_ice_cloud_optics  = command_line_switches.at("ice-cloud-optics"  ).first;
    const bool switch_multi       = command_line_switches.at("multi-start-points").first;
    const bool switch_compress    = command_line_switches.at("compress").first;
    const bool switch_compress_bkg = command_line_switches.at("compress-bkg").first;
    const bool switch_tilt_sza             = command_line_switches.at("tilt-sza"    ).first;
    const bool switch_tilt_azi             = command_line_switches.at("tilt-azi"    ).first;

    if (switch_cloud_optics)
    {
        switch_liq_cloud_optics = true;
        switch_ice_cloud_optics = true;
    }
    if (switch_liq_cloud_optics || switch_ice_cloud_optics)
    {
        switch_cloud_optics = true;
    }
    if (switch_tilt_sza && !switch_tilt_azi) {
        std::string error = "If tilt-sza is provided, user must pass tilt-azi too.";
        throw std::runtime_error(error);
    }

    // Print the options to the screen.
    print_command_line_options(command_line_switches, command_line_ints);

    int n_points_sqrt = 1;
    if (switch_multi) 
    {
        n_points_sqrt = Int(command_line_ints.at("multi-start-points").first);
    }

    Float sza = 0;
    Float azi = 0;
    if (switch_tilt_sza) 
    {
        int sza_deg = Int(command_line_ints.at("tilt-sza").first);
        sza = sza_deg * 3.14159f / 180.0f;
    }
    if (switch_tilt_azi) 
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

    const int n_lay_init = input_nc.get_dimension_size("lay");
    const int n_lev_init = input_nc.get_dimension_size("lev");
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
    Array<Float,2> p_lay(input_nc.get_variable<Float>("p_lay", {n_lay_init, n_col_y, n_col_x}), {n_col, n_lay_init});
    Array<Float,2> t_lay(input_nc.get_variable<Float>("t_lay", {n_lay_init, n_col_y, n_col_x}), {n_col, n_lay_init});
    Array<Float,2> p_lev(input_nc.get_variable<Float>("p_lev", {n_lev_init, n_col_y, n_col_x}), {n_col, n_lev_init});
    Array<Float,2> t_lev(input_nc.get_variable<Float>("t_lev", {n_lev_init, n_col_y, n_col_x}), {n_col, n_lev_init});

    // Fetch the col_dry in case present.
    Array<Float,2> col_dry;
    if (input_nc.variable_exists("col_dry"))
    {
        col_dry.set_dims({n_col, n_lay_init});
        col_dry = std::move(input_nc.get_variable<Float>("col_dry", {n_lay_init, n_col_y, n_col_x}));
    }

    // Create container for the gas concentrations and read gases.
    Gas_concs gas_concs;

    read_and_set_vmr("h2o", n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("co2", n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("o3" , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("n2o", n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("co" , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("ch4", n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("o2" , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("n2" , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);

    read_and_set_vmr("ccl4"   , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("cfc11"  , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("cfc12"  , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("cfc22"  , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("hfc143a", n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("hfc125" , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("hfc23"  , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("hfc32"  , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("hfc134a", n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("cf4"    , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);
    read_and_set_vmr("no2"    , n_col_x, n_col_y, n_lay_init, input_nc, gas_concs);

    Array<Float,2> lwp;
    Array<Float,2> iwp;
    Array<Float,2> rel;
    Array<Float,2> dei;

    if (switch_cloud_optics)
    {

        if (switch_liq_cloud_optics)
        {
            lwp.set_dims({n_col, n_lay_init});
            lwp = std::move(input_nc.get_variable<Float>("lwp", {n_lay_init, n_col_y, n_col_x}));

            rel.set_dims({n_col, n_lay_init});
            rel = std::move(input_nc.get_variable<Float>("rel", {n_lay_init, n_col_y, n_col_x}));
        }

        if (switch_ice_cloud_optics)
        {
            iwp.set_dims({n_col, n_lay_init});
            iwp = std::move(input_nc.get_variable<Float>("iwp", {n_lay_init, n_col_y, n_col_x}));

            dei.set_dims({n_col, n_lay_init});
            dei = std::move(input_nc.get_variable<Float>("dei", {n_lay_init, n_col_y, n_col_x}));
        }
    }
    ////// FINISH READING THE ATMOSPHERIC DATA //////

    // if t lev all 0, interpolate from t lay
    if (*std::max_element(t_lev.v().begin(), t_lev.v().end()) <= 0) {
        for (int i = 1; i <= n_col; ++i) {
            for (int j = 2; j <= n_lay_init; ++j) {
                t_lev({i, j}) = (t_lay({i, j}) + t_lay({i, j - 1})) / 2.0;
            }
            t_lev({i, n_lev_init}) = 2 * t_lay({i, n_lay_init}) - t_lev({i,n_lay_init});
            t_lev({i, 1}) = 2 * t_lay({i, 1}) - t_lev({i,2});
        }
    }

    int n_lay_HOLD;
    int n_lev_HOLD;
    const int bkg_start_idx = n_z_in;
    int start_idx;
    int bkg_reduction_len;
    if (switch_compress_bkg)
    {
        if ((n_lay_init - n_z_in) % 2 != 0) {
            start_idx = n_z_in + 1;
        }
        else
        {
            start_idx = n_z_in;
        }
        bkg_reduction_len = (n_lay_init - start_idx)/2;
        int n_lay_new = start_idx + (n_lay_init - start_idx)/2;
        int n_lev_new = n_lay_new + 1.0;

        Status::print_message("Compress Background Profile.");
        if (switch_liq_cloud_optics)
        {
            compress_columns_weighted_avg(n_col_x, n_col_y, 
                                            n_lay_new, start_idx, 
                                            rel.v(), lwp.v());
            rel.expand_dims({n_col, n_lay_new});

            compress_columns(n_col_x, n_col_y, 
                                n_lay_new, 
                                start_idx, lwp.v());
            lwp.expand_dims({n_col, n_lay_new}); 
        }
        if (switch_ice_cloud_optics)
        {
            compress_columns_weighted_avg(n_col_x, n_col_y, 
                                            n_lay_new, start_idx, 
                                            dei.v(), iwp.v());
            dei.expand_dims({n_col, n_lay_new});
            
            compress_columns(n_col_x, n_col_y, 
                                n_lay_new, 
                                start_idx, iwp.v());
            iwp.expand_dims({n_col, n_lay_new}); 
        }


        for (const auto& gas_name : gas_names) {
            if (!gas_concs.exists(gas_name)) {
                continue;
            }
            const Array<Float,2>& gas = gas_concs.get_vmr(gas_name);
            std::string var_name = "vmr_" + gas_name;
            if (gas.size() > 1) {
                Array<Float,2> gas_tmp(gas);
                compress_columns_weighted_avg(n_col_x, n_col_y,
                                                n_lay_new, 
                                                start_idx, 
                                                gas_tmp.v(), 
                                                p_lay.v());
                gas_tmp.expand_dims({n_col, n_lay_new});
                gas_concs.set_vmr(gas_name, gas_tmp);
            }
        } 

        compress_columns_p_or_t(n_col_x, n_col_y, n_lay_new, 
                                start_idx, 
                                p_lev.v(), p_lay.v());
        p_lay.expand_dims({n_col, n_lay_new});
        p_lev.expand_dims({n_col, n_lev_new});
        compress_columns_p_or_t(n_col_x, n_col_y, n_lay_new, 
                                start_idx, 
                                t_lev.v(), t_lay.v());
        t_lay.expand_dims({n_col, n_lay_new});
        t_lev.expand_dims({n_col, n_lev_new});

        n_lay_HOLD = n_lay_new;
        n_lev_HOLD = n_lev_new;

        Status::print_message("Finish background compression.");

    } else{
        n_lay_HOLD = n_lay_init;
        n_lev_HOLD = n_lev_init;
    }
    const int n_lay = n_lay_HOLD;
    const int n_lev = n_lev_HOLD;

    ////// SETUP FOR TILTING //////

    std::vector<std::pair<Float, Float>> x_y_start_arr(n_points_sqrt*n_points_sqrt);
    Float d = 1.0f / n_points_sqrt;
    int idx = 0;
    for (int i = 0; i < n_points_sqrt; i++){
        for (int j = 0; j < n_points_sqrt; j++){
            x_y_start_arr[idx++] = std::make_pair(i*d+d/2, j*d+d/2);
        }
    }

    Float x_start;
    Float y_start;

    std::vector<Float> z_out;
    std::vector<Float> zh_out;

    std::string file_name;
    int n_z_tilt;
    int n_zh_tilt;

    std::vector<Float> z_out_compress;
    std::vector<Float> zh_out_compress;
    int n_lev_compress;
    int n_lay_compress;
    int compress_lay_start_idx;

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
            Status::print_message("Multiple Start Points Unnecessary with Sun Overhead.");
            break;
        }
        loop_index++;
        file_name = "rte_rrtmgp_input_" + std::to_string(loop_index) + ".nc";
        x_start = pair.first;
        y_start = pair.second;

        std::cout << "x_start: " << x_start << std::endl;
        std::cout << "y_start: " << y_start << std::endl;

        Status::print_message("###### Starting Tilting ######");
        auto time_start = std::chrono::high_resolution_clock::now();
        
        tilted_path(xh.v(),yh.v(),zh.v(),z.v(),sza,azi,x_start, y_start, path.v(),zh_tilt.v());
        std::cout << "finish tilted path" << std::endl;

        n_zh_tilt = zh_tilt.v().size();
        n_z_tilt = n_zh_tilt - 1;
        if (loop_index == 1) {  
            z_out = linspace(z.v()[0], z.v()[n_z_in - 1], n_z_tilt);
            zh_out = linspace(zh.v()[0], zh.v()[n_zh_in - 1], n_zh_tilt);

            if (switch_compress){

                int idx_hold = 2*(n_z_tilt - n_z_in);
                if ((z_out.size() - idx_hold) % 2 != 0) {
                    idx_hold--;
                }

                n_lay_compress = (n_z_tilt - idx_hold) + (idx_hold)/2;
                n_lev_compress = n_lay_compress + 1;
                compress_lay_start_idx = (n_z_tilt - idx_hold);
                std::cout << "n_z_tilt: " << n_z_tilt << " idx_hold: " << idx_hold << " compress_lay_start_idx: "<< compress_lay_start_idx << std::endl;
                if (switch_compress_bkg) {
                    compress_lay_start_idx = compress_lay_start_idx + bkg_reduction_len;
                }
                std::cout << "compress_lay_start_idx: "<< compress_lay_start_idx << std::endl;
                if (compress_lay_start_idx < 0) {
                    throw std::runtime_error("compress_lay_start_idx is negative - SZA too high.");
                }

                z_out_compress = linspace(z.v()[0], z.v()[n_z_in - 1], n_lay_compress);
                zh_out_compress = linspace(zh.v()[0], zh.v()[n_zh_in - 1], n_lev_compress);
                assert(n_lev_compress == n_zh_in && n_lay_compress == n_z_in);
            }
        }

        
        
        path.set_dims({n_z_tilt}); 
        zh_tilt.set_dims({n_zh_tilt}); 
        if (switch_cloud_optics)
        {
            for (int ilay=1; ilay<=n_zh_in; ++ilay)    
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
                lwp_copy.expand_dims({n_col, n_z_tilt});
                rel_copy.expand_dims({n_col, n_z_tilt});
            }
            if (switch_ice_cloud_optics)
            {
                create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), iwp_copy.v());
                create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), dei_copy.v());

                iwp_copy.expand_dims({n_col, n_z_tilt});
                dei_copy.expand_dims({n_col, n_z_tilt});
            }

            for (int ilay=1; ilay<=n_z_tilt; ++ilay)    
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

            if (gas.size() > 1) {
                Array<Float,2> gas_tmp(gas);
                create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), gas_tmp.v());
                gas_tmp.expand_dims({n_col, n_z_tilt});
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
        create_tilted_columns_levlay(n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), t_lay_copy.v(), t_lev_copy.v());
        create_tilted_columns_levlay(n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), p_lay_copy.v(), p_lev_copy.v());
        t_lay_copy.expand_dims({n_col, n_z_tilt});
        t_lev_copy.expand_dims({n_col, n_zh_tilt});
        p_lay_copy.expand_dims({n_col, n_z_tilt});
        p_lev_copy.expand_dims({n_col, n_zh_tilt});

        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(time_end-time_start).count();

        Status::print_message("Duration tilting columns: " + std::to_string(duration) + " (ms)");
        if (switch_compress){
            Status::print_message("Compress.");
            if (switch_liq_cloud_optics)
            {
                compress_columns_weighted_avg(n_col_x, n_col_y, 
                                                n_z_in, compress_lay_start_idx, 
                                                rel_copy.v(), lwp_copy.v());
                rel_copy.expand_dims({n_col, n_z_in});

                compress_columns(n_col_x, n_col_y, 
                                    n_z_in, 
                                    compress_lay_start_idx, lwp_copy.v());
                lwp_copy.expand_dims({n_col, n_z_in}); 
            }
            if (switch_ice_cloud_optics)
            {
                compress_columns_weighted_avg(n_col_x, n_col_y, 
                                                n_z_in, compress_lay_start_idx, 
                                                dei_copy.v(), iwp_copy.v());
                dei_copy.expand_dims({n_col, n_z_in});
                
                compress_columns(n_col_x, n_col_y, 
                                    n_z_in, 
                                    compress_lay_start_idx, iwp_copy.v());
                iwp_copy.expand_dims({n_col, n_z_in}); 
            }


            for (const auto& gas_name : gas_names) {
                if (!gas_concs_copy.exists(gas_name)) {
                    continue;
                }
                const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);
                std::string var_name = "vmr_" + gas_name;
                if (gas.size() > 1) {
                    Array<Float,2> gas_tmp(gas);
                    compress_columns_weighted_avg(n_col_x, n_col_y,
                                                    n_z_in, 
                                                    compress_lay_start_idx, 
                                                    gas_tmp.v(), 
                                                    p_lay_copy.v());
                    gas_tmp.expand_dims({n_col, n_z_in});
                    gas_concs_copy.set_vmr(gas_name, gas_tmp);
                }
            } 

            compress_columns_p_or_t(n_col_x, n_col_y, n_z_in, 
                                    compress_lay_start_idx, 
                                    p_lev_copy.v(), p_lay_copy.v());
            p_lay_copy.expand_dims({n_col, n_z_in});
            p_lev_copy.expand_dims({n_col, n_zh_in});
            compress_columns_p_or_t(n_col_x, n_col_y, n_z_in, 
                                    compress_lay_start_idx, 
                                    t_lev_copy.v(), t_lay_copy.v());
            t_lay_copy.expand_dims({n_col, n_z_in});
            t_lev_copy.expand_dims({n_col, n_zh_in});

            Status::print_message("Finish z grid compression.");

            Status::print_message("Add background profile back on.");
            if (switch_liq_cloud_optics)
            {
                restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, n_z_in, lwp_copy.v(), lwp.v());
                lwp_copy.expand_dims({n_col, n_lay});
                restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, n_z_in, rel_copy.v(), rel.v());
                rel_copy.expand_dims({n_col, n_lay});
            }
            if (switch_ice_cloud_optics)
            {
                restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, n_z_in, iwp_copy.v(), iwp.v());
                iwp_copy.expand_dims({n_col, n_lay});
                restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, n_z_in, dei_copy.v(), dei.v());
                dei_copy.expand_dims({n_col, n_lay});
            }

            for (const auto& gas_name : gas_names) {
                if (!gas_concs_copy.exists(gas_name)) {
                    continue;
                }
                const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);
                const Array<Float,2>& gas_full = gas_concs.get_vmr(gas_name);

                std::string var_name = "vmr_" + gas_name;
                if (gas.size() > 1) {
                    std::vector<Float> gas_copy = gas.v();
                    std::vector<Float> gas_full_copy = gas_full.v();
                    restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, n_z_in, gas_copy, gas_full_copy);
                    
                    Array<Float,2> gas_tmp({n_col, n_lay});
                    gas_tmp = std::move(gas_copy);
                    gas_tmp.expand_dims({n_col, n_lay});
                    
                    gas_concs_copy.set_vmr(gas_name, gas_tmp);
                    
                }
            }

            restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, n_z_in, p_lay_copy.v(), p_lay.v());
            p_lay_copy.expand_dims({n_col, n_lay});
            restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, n_z_in, t_lay_copy.v(), t_lay.v());
            t_lay_copy.expand_dims({n_col, n_lay});

            restore_bkg_profile(n_col_x, n_col_y, n_lev, n_zh_in, n_zh_in, p_lev_copy.v(), p_lev.v());
            p_lev_copy.expand_dims({n_col, n_lev});
            restore_bkg_profile(n_col_x, n_col_y, n_lev, n_zh_in, n_zh_in, t_lev_copy.v(), t_lev.v());
            t_lev_copy.expand_dims({n_col, n_lev});

            Status::print_message("prepare_netcdf.");
            prepare_netcdf(input_nc, file_name, n_lay, n_lev, n_col_x, n_col_y, n_zh_in, n_z_in,
                    sza, zh_out_compress, z_out_compress,
                    p_lay_copy, t_lay_copy, p_lev_copy, t_lev_copy, 
                    lwp_copy, iwp_copy, rel_copy, dei_copy, 
                    gas_concs_copy, gas_names, 
                    switch_cloud_optics, switch_liq_cloud_optics, switch_ice_cloud_optics);

        }
        else {

            Status::print_message("Add background profile back on.");

            // add background profile back on
            const int n_lay_tot = n_z_tilt + (n_lay - bkg_start_idx);
            const int n_lev_tot = n_lay_tot + 1;

            if (switch_liq_cloud_optics)
            {
                restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_tilt, bkg_start_idx, lwp_copy.v(), lwp.v());
                lwp_copy.expand_dims({n_col, n_lay_tot});
                

                restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_tilt, bkg_start_idx, rel_copy.v(), rel.v());
                rel_copy.expand_dims({n_col, n_lay_tot});
            }
            if (switch_ice_cloud_optics)
            {
                restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_tilt, bkg_start_idx, iwp_copy.v(), iwp.v());
                iwp_copy.expand_dims({n_col, n_lay_tot});
                restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_tilt, bkg_start_idx, dei_copy.v(), dei.v());
                dei_copy.expand_dims({n_col, n_lay_tot});
            }

            for (const auto& gas_name : gas_names) {
                if (!gas_concs_copy.exists(gas_name)) {
                    continue;
                }
                const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);
                const Array<Float,2>& gas_full = gas_concs.get_vmr(gas_name);

                std::string var_name = "vmr_" + gas_name;
                if (gas.size() > 1) {
                    std::vector<Float> gas_copy = gas.v();
                    std::vector<Float> gas_full_copy = gas_full.v();

                    restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_tilt, bkg_start_idx, gas_copy, gas_full_copy);
                    

                    Array<Float,2> gas_tmp({n_col, n_lay});
                    gas_tmp = std::move(gas_copy);
                    gas_tmp.expand_dims({n_col, n_lay_tot});
                    
                    gas_concs_copy.set_vmr(gas_name, gas_tmp);
                    
                }
            }

            restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_tilt, bkg_start_idx, p_lay_copy.v(), p_lay.v());
            p_lay_copy.expand_dims({n_col, n_lay_tot});
            restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_tilt, bkg_start_idx, t_lay_copy.v(), t_lay.v());
            t_lay_copy.expand_dims({n_col, n_lay_tot});

            restore_bkg_profile(n_col_x, n_col_y, n_lev, n_zh_tilt, (bkg_start_idx + 1), p_lev_copy.v(), p_lev.v());
            p_lev_copy.expand_dims({n_col, n_lev_tot});
            restore_bkg_profile(n_col_x, n_col_y, n_lev, n_zh_tilt, (bkg_start_idx + 1), t_lev_copy.v(), t_lev.v());
            t_lev_copy.expand_dims({n_col, n_lev_tot});

            Status::print_message("prepare_netcdf.");
            prepare_netcdf(input_nc, file_name, n_lay_tot, n_lev_tot, n_col_x, n_col_y, n_zh_tilt, n_z_tilt, 
                        sza, zh_out, z_out,
                        p_lay_copy, t_lay_copy, p_lev_copy, t_lev_copy, 
                        lwp_copy, iwp_copy, rel_copy, dei_copy, 
                        gas_concs_copy, gas_names, 
                        switch_cloud_optics, switch_liq_cloud_optics, switch_ice_cloud_optics);
        }    
    }

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
