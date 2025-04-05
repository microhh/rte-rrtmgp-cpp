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
 #include "tilt_utils.h"
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

    for (const auto& gas_name : gas_names) {
        if (!gas_concs.exists(gas_name)) {
            continue;
        }
        const Array<Float,2>& gas = gas_concs.get_vmr(gas_name);
        std::string var_name = "vmr_" + gas_name;
        if ((gas.size() > 1) && (gas_name != "h2o" && gas_name != "o3")) {
            throw std::runtime_error("3D gas concentrations only supported for h20 and o3.");
        }
    }

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
    ////// FINISH READING THE ATMOSPHERIC DATA //////

    Array<Float,2> t_lay_out = t_lay;
    Array<Float,2> t_lev_out = t_lev;
    Array<Float,2> p_lay_out = p_lay;
    Array<Float,2> p_lev_out = p_lev;
    Gas_concs gas_concs_out = gas_concs;

    Array<Float,2> lwp_out;
    lwp_out.set_dims({n_col, n_z_in});
    Array<Float,2> rel_out;
    rel_out.set_dims({n_col, n_z_in});
    Array<Float,2> iwp_out;
    iwp_out.set_dims({n_col, n_z_in});
    Array<Float,2> dei_out;
    dei_out.set_dims({n_col, n_z_in});
 
    tica_tilt(
        sza, azi,
        n_col_x, n_col_y, n_col,
        n_lay, n_lev, n_z_in, n_zh_in ,
        xh, yh, zh, z,
        p_lay, t_lay, p_lev, t_lev, 
        lwp, iwp, rel, dei, 
        gas_concs,
        p_lay_out, t_lay_out, p_lev_out, t_lev_out, 
        lwp_out, iwp_out, rel_out, dei_out, 
        gas_concs_out, 
        gas_names,
        switch_cloud_optics, switch_liq_cloud_optics, switch_ice_cloud_optics
    );
     
    std::vector<Float> z_out_compress = linspace(z.v()[0], z.v()[n_z_in - 1], n_z_in);
    std::vector<Float> zh_out_compress = linspace(zh.v()[0], zh.v()[n_zh_in - 1], n_zh_in);

    Status::print_message("Prepare Netcdf.");
    std::string file_name = "test.nc";
    prepare_netcdf(input_nc, file_name, n_lay, n_lev, n_col_x, n_col_y, n_zh_in, n_z_in,
        sza, zh_out_compress, z_out_compress,
        p_lay_out, t_lay_out, p_lev_out, t_lev_out, 
        lwp_out, iwp_out, rel_out, dei_out, 
        gas_concs_out, gas_names, 
        switch_cloud_optics, switch_liq_cloud_optics, switch_ice_cloud_optics);
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
