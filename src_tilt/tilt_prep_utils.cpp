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
#include "tilt_prep_utils.h"
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

void tilt_fields(const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1> zh, const Array<Float,1> z,
    const Array<Float,1> zh_tilt, const Array<ijk,1> path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Gas_concs& gas_concs_copy, const std::vector<std::string> gas_names,
    const bool switch_cloud_optics, const bool switch_liq_cloud_optics, const bool switch_ice_cloud_optics
) {
    if (switch_cloud_optics)
    {
        for (int ilay = 1; ilay <= n_zh_in; ++ilay)    
        {
            Float dz = zh({ilay + 1}) - zh({ilay});
            for (int icol = 1; icol <= n_col; ++icol)    
            {
                if (switch_liq_cloud_optics)
                {
                    (*lwp_copy)({icol, ilay}) /= dz;
                }
                if (switch_ice_cloud_optics)
                {
                    (*iwp_copy)({icol, ilay}) /= dz;
                }
            }
        }
        if (switch_liq_cloud_optics)
        {
            create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), lwp_copy->v());
            create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), rel_copy->v());

            lwp_copy->expand_dims({n_col, n_z_tilt});
            rel_copy->expand_dims({n_col, n_z_tilt});
        }
        if (switch_ice_cloud_optics)
        {
            create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), iwp_copy->v());
            create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), dei_copy->v());

            iwp_copy->expand_dims({n_col, n_z_tilt});
            dei_copy->expand_dims({n_col, n_z_tilt});
        }

        for (int ilay = 1; ilay <= n_z_tilt; ++ilay)    
        {
            Float dz = zh_tilt({ilay + 1}) - zh_tilt({ilay});
            for (int icol = 1; icol <= n_col; ++icol)    
            {
                if (switch_liq_cloud_optics)
                {
                    (*lwp_copy)({icol, ilay}) *= dz;
                }
                if (switch_ice_cloud_optics)
                {
                    (*iwp_copy)({icol, ilay}) *= dz;
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

    // Create tilted columns for T and p. Important: create T first!!
    create_tilted_columns_levlay(n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), t_lay_copy->v(), t_lev_copy->v());
    create_tilted_columns_levlay(n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), p_lay_copy->v(), p_lev_copy->v());

    t_lay_copy->expand_dims({n_col, n_z_tilt});
    t_lev_copy->expand_dims({n_col, n_zh_tilt});
    p_lay_copy->expand_dims({n_col, n_z_tilt});
    p_lev_copy->expand_dims({n_col, n_zh_tilt});
}

void tilt_fields_single_column(const int idx_col_x, const int idx_col_y,
    const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1> zh, const Array<Float,1> z,
    const Array<Float,1> zh_tilt, const Array<ijk,1> path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Gas_concs& gas_concs_copy, const std::vector<std::string> gas_names,
    const bool switch_cloud_optics, const bool switch_liq_cloud_optics, const bool switch_ice_cloud_optics
) {
    if (switch_cloud_optics)
    {
        for (int ilay = 1; ilay <= n_zh_in; ++ilay)    
        {
            Float dz = zh({ilay + 1}) - zh({ilay});
            for (int icol = 1; icol <= n_col; ++icol)    
            {
                if (switch_liq_cloud_optics)
                {
                    (*lwp_copy)({icol, ilay}) /= dz;
                }
                if (switch_ice_cloud_optics)
                {
                    (*iwp_copy)({icol, ilay}) /= dz;
                }
            }
        }
        if (switch_liq_cloud_optics)
        {
            create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), lwp_copy->v());
            create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), rel_copy->v());

            lwp_copy->expand_dims({1, n_z_tilt});
            rel_copy->expand_dims({1, n_z_tilt});
        }
        if (switch_ice_cloud_optics)
        {
            create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), iwp_copy->v());
            create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), dei_copy->v());

            iwp_copy->expand_dims({1, n_z_tilt});
            dei_copy->expand_dims({1, n_z_tilt});
        }

        for (int ilay = 1; ilay <= n_z_tilt; ++ilay)    
        {
            Float dz = zh_tilt({ilay + 1}) - zh_tilt({ilay});
            for (int icol = 1; icol <= n_col; ++icol)    
            {
                if (switch_liq_cloud_optics)
                {
                    (*lwp_copy)({icol, ilay}) *= dz;
                }
                if (switch_ice_cloud_optics)
                {
                    (*iwp_copy)({icol, ilay}) *= dz;
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
            create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), gas_tmp.v());
            gas_tmp.expand_dims({1, n_z_tilt});
            gas_concs_copy.set_vmr(gas_name, gas_tmp);
        } 
        else if (gas.size() == 1) {
            // Do nothing for single profiles
        } 
        else {
            throw std::runtime_error("No tilted column implementation for single profiles.");
        }
    }

    // Create tilted columns for T and p. Important: create T first!!
    create_single_tilted_columns_levlay(idx_col_x, idx_col_y,  n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), t_lay_copy->v(), t_lev_copy->v());
    create_single_tilted_columns_levlay(idx_col_x, idx_col_y,  n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), p_lay_copy->v(), p_lev_copy->v());

    t_lay_copy->expand_dims({1, n_z_tilt});
    t_lev_copy->expand_dims({1, n_zh_tilt});
    p_lay_copy->expand_dims({1, n_z_tilt});
    p_lev_copy->expand_dims({1, n_zh_tilt});
}

void compress_fields(const int compress_lay_start_idx, const int n_col_x, const int n_col_y,
    const int n_z_in, const int n_zh_in,  const int n_z_tilt,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Gas_concs& gas_concs_copy, std::vector<std::string> gas_names,
    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics)
{
    const int n_col = n_col_x*n_col_y;

    if (switch_liq_cloud_optics)
    {
        compress_columns_weighted_avg(n_col_x, n_col_y, 
                                        n_z_in, n_z_tilt, compress_lay_start_idx, 
                                        rel_copy->v(), lwp_copy->v());
        rel_copy->expand_dims({n_col, n_z_in});

        compress_columns(n_col_x, n_col_y, 
                            n_z_in, n_z_tilt,
                            compress_lay_start_idx, lwp_copy->v());
        lwp_copy->expand_dims({n_col, n_z_in}); 
    }
    if (switch_ice_cloud_optics)
    {
        compress_columns_weighted_avg(n_col_x, n_col_y, 
                                        n_z_in, n_z_tilt, compress_lay_start_idx, 
                                        dei_copy->v(), iwp_copy->v());
        dei_copy->expand_dims({n_col, n_z_in});
        
        compress_columns(n_col_x, n_col_y, 
                            n_z_in, n_z_tilt,
                            compress_lay_start_idx, iwp_copy->v());
        iwp_copy->expand_dims({n_col, n_z_in}); 
    }

    for (const auto& gas_name : gas_names) {
        if (!gas_concs_copy.exists(gas_name)) {
            continue;
        }
        const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);
        if (gas.size() > 1) {
            Array<Float,2> gas_tmp(gas);
            compress_columns_weighted_avg(n_col_x, n_col_y,
                                            n_z_in, n_z_tilt,
                                            compress_lay_start_idx, 
                                            gas_tmp.v(), 
                                            p_lay_copy->v());
            gas_tmp.expand_dims({n_col, n_z_in});
            gas_concs_copy.set_vmr(gas_name, gas_tmp);
        }
    }

    compress_columns_p_or_t(n_col_x, n_col_y, n_z_in, n_z_tilt,
                            compress_lay_start_idx, 
                            p_lev_copy->v(), p_lay_copy->v());
    p_lay_copy->expand_dims({n_col, n_z_in});
    p_lev_copy->expand_dims({n_col, n_zh_in});
    compress_columns_p_or_t(n_col_x, n_col_y, n_z_in, n_z_tilt,
                            compress_lay_start_idx, 
                            t_lev_copy->v(), t_lay_copy->v());
    t_lay_copy->expand_dims({n_col, n_z_in});
    t_lev_copy->expand_dims({n_col, n_zh_in});
}

void restore_bkg_profile(const int n_x, const int n_y, 
                      const int n_full,
                      const int n_tilt,
                      const int bkg_start, 
                      std::vector<Float>& var,
                      std::vector<Float>& var_w_bkg)
{
    const int n_out = n_tilt + (n_full - bkg_start);

    std::vector<Float> var_tmp(n_out * n_x * n_y);    
    #pragma omp parallel for
    for (int ilay = 0; ilay < n_tilt; ++ilay)
    {        
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                var_tmp[out_idx] = var[out_idx];
            }
        }
    }

    #pragma omp parallel for
    for (int ilay = n_tilt; ilay < n_out; ++ilay)
    {
        int ilay_in = ilay - (n_tilt - bkg_start);
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                const int in_idx = ix + iy * n_x + ilay_in * n_x * n_y;
                var_tmp[out_idx] = var_w_bkg[in_idx];
            }
        }
    }
    var = var_tmp;
}

void restore_bkg_profile_bundle(const int n_col_x, const int n_col_y, 
    const int n_lay, const int n_lev, 
    const int n_lay_tot, const int n_lev_tot, 
    const int n_z_in, const int n_zh_in,
    const int bkg_start_z, const int bkg_start_zh, 
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Gas_concs& gas_concs_copy,
    Array<Float,2>* p_lay, Array<Float,2>* t_lay, Array<Float,2>* p_lev, Array<Float,2>* t_lev, 
    Array<Float,2>* lwp, Array<Float,2>* iwp, Array<Float,2>* rel, Array<Float,2>* dei, 
    Gas_concs& gas_concs, 
    std::vector<std::string> gas_names,
    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics
)
{
    const int n_col = n_col_x*n_col_y;

    if (switch_liq_cloud_optics)
    {
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, lwp_copy->v(), lwp->v());
        lwp_copy->expand_dims({n_col, n_lay_tot});
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, rel_copy->v(), rel->v());
        rel_copy->expand_dims({n_col, n_lay_tot});
    }
    if (switch_ice_cloud_optics)
    {
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, iwp_copy->v(), iwp->v());
        iwp_copy->expand_dims({n_col, n_lay_tot});
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, dei_copy->v(), dei->v());
        dei_copy->expand_dims({n_col, n_lay_tot});
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
            restore_bkg_profile(n_col_x, n_col_y, 
                                n_lay, n_z_in, 
                                bkg_start_z, 
                                gas_copy, gas_full_copy);
            
            Array<Float,2> gas_tmp({n_col, n_lay_tot});
            gas_tmp = std::move(gas_copy);
            gas_tmp.expand_dims({n_col, n_lay_tot});
            
            gas_concs_copy.set_vmr(gas_name, gas_tmp);
            
        }
    }

    restore_bkg_profile(n_col_x, n_col_y, 
                        n_lay, n_z_in, 
                        bkg_start_z, 
                        p_lay_copy->v(), p_lay->v());
    p_lay_copy->expand_dims({n_col, n_lay_tot});

    restore_bkg_profile(n_col_x, n_col_y, 
                        n_lay, n_z_in, 
                        bkg_start_z, 
                        t_lay_copy->v(), t_lay->v());
    t_lay_copy->expand_dims({n_col, n_lay_tot});

    restore_bkg_profile(n_col_x, n_col_y, 
                        n_lev, n_zh_in, 
                        bkg_start_zh, 
                        p_lev_copy->v(), p_lev->v());
    p_lev_copy->expand_dims({n_col, n_lev_tot});
    restore_bkg_profile(n_col_x, n_col_y, 
                        n_lev, n_zh_in, 
                        bkg_start_zh, 
                        t_lev_copy->v(), t_lev->v());
    t_lev_copy->expand_dims({n_col, n_lev_tot});
}


void post_process_output(const std::vector<ColumnResult>& col_results,
                         const int n_col_x, const int n_col_y,
                         const int n_z, const int n_zh,
                         Array<Float,2>* p_lay_out,
                         Array<Float,2>* t_lay_out,
                         Array<Float,2>* p_lev_out,
                         Array<Float,2>* t_lev_out,
                         Array<Float,2>* lwp_out,
                         Array<Float,2>* iwp_out,
                         Array<Float,2>* rel_out,
                         Array<Float,2>* dei_out,
                         Gas_concs& gas_concs_out,
                         const std::vector<std::string>& gas_names,
                         const bool switch_cloud_optics,
                         const bool switch_liq_cloud_optics,
                         const bool switch_ice_cloud_optics)
{
    const int total_cols = n_col_x * n_col_y;
    const int stride = total_cols;
    for (int idx_y = 0; idx_y < n_col_y; ++idx_y) {
        for (int idx_x = 0; idx_x < n_col_x; ++idx_x) {
            int col_idx = idx_x + idx_y * n_col_x;
            const ColumnResult& col = col_results[col_idx];
            int base_idx = col_idx;

            for (int j = 0; j < n_z; ++j) {
                int out_idx = base_idx + j * stride;
                p_lay_out->v()[out_idx] = col.p_lay.v()[j];
                t_lay_out->v()[out_idx] = col.t_lay.v()[j];
                if (switch_liq_cloud_optics) {
                    lwp_out->v()[out_idx] = col.lwp.v()[j];
                    rel_out->v()[out_idx] = col.rel.v()[j];
                }
                if (switch_ice_cloud_optics) {
                    iwp_out->v()[out_idx] = col.iwp.v()[j];
                    dei_out->v()[out_idx] = col.dei.v()[j];
                }
            }

            for (int j = 0; j < n_zh; ++j) {
                int out_idx = base_idx + j * stride;
                p_lev_out->v()[out_idx] = col.p_lev.v()[j];
                t_lev_out->v()[out_idx] = col.t_lev.v()[j];
            }
            
            for (const auto& gas_name : gas_names) {
                if (!col.gas_concs.exists(gas_name))
                    continue;
                const Array<Float,2>& gas_src = col.gas_concs.get_vmr(gas_name);
                if (gas_src.size() > 1) {
                    Array<Float,2>& gas_dest = const_cast<Array<Float,2>&>(gas_concs_out.get_vmr(gas_name));
                    for (int j = 0; j < n_z; ++j) {
                        const int out_idx = base_idx + j * stride;
                        gas_dest.v()[out_idx] = gas_src.v()[j];
                    }
                }
            }
        }
    }
}

