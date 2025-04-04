#include <cmath>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "Gas_concs.h"
#include "tilt_utils.h"
#include "types.h"

void tilted_path(std::vector<Float>& xh, std::vector<Float>& yh,
                 std::vector<Float>& zh, std::vector<Float>& z,
                 Float sza, Float azi,
                 Float x_start, Float y_start,
                 std::vector<ijk>& tilted_path,
                 std::vector<Float>& zh_tilted)
{
    const Float dx = xh[1]-xh[0]; 
    const Float dy = yh[1]-yh[0]; 
    const Float z_top = *std::max_element(zh.begin(), zh.end());
    const int n_x = xh.size()-1;
    const int n_y = yh.size()-1;
    int i = 0;
    int j = 0;
    int k = 0;

    Float xp = xh[0] + x_start*dx;
    Float yp = yh[0] + y_start*dy;
    Float zp = 0.;
    Float dl = 0.;

    tilted_path.clear();
    zh_tilted.clear();
    std::vector<Float> dz_tilted;

    Float dir_x = std::sin(sza) * std::sin(azi); // azi 0 is from the north
    Float dir_y = std::sin(sza) * std::cos(azi);
    Float dir_z = std::cos(sza);  

    int z_idx = -1;
    
    const Float epsilon = 1e-8; // Small value to handle floating-point precision
    const Float min_step = 1e-2; // Minimum step size in meters
    
    tilted_path.push_back({i, j, k}); // Add starting point
    dz_tilted.push_back(0.0);
    z_idx = 0;
    
    while (zp < z_top)
    {           
        // Check bounds before accessing arrays
        if (k + 1 >= zh.size()) {
            std::cerr << "Error: k+1 (" << k + 1 << ") out of bounds for zh (size=" << zh.size() << ")" << std::endl;
            break;
        }
        if (j + 1 >= yh.size()) {
            std::cerr << "Error: j+1 (" << j + 1 << ") out of bounds for yh (size=" << yh.size() << ")" << std::endl;
            break;
        }
        if (i + 1 >= xh.size()) {
            std::cerr << "Error: i+1 (" << i + 1 << ") out of bounds for xh (size=" << xh.size() << ")" << std::endl;
            break;
        }
        
        // Calculate distances to next cell boundaries
        Float lz = (std::abs(dir_z) < epsilon) ? 
                    100000.: (zh[k+1] - zp) / dir_z;
        
        // Handle cases where we're extremely close to a boundary
        if (std::abs(zp - zh[k+1]) < epsilon && dir_z > 0) {
            k += 1;
            zp = zh[k];
            if (k + 1 >= zh.size()) {
                break; // We've reached the top
            }
            continue;
        }
        
        Float ly;
        if (std::abs(dir_y) < epsilon) {
            ly = 100000.;
        } else if (dir_y < 0) {
            if (std::abs(yp - yh[j]) < epsilon) {
                // Already at the lower boundary, move to the previous cell
                j = (j == 0) ? n_y - 1 : j - 1;
                yp = yh[j+1] - epsilon; // Position just inside the cell
                continue;
            }
            ly = (yp - yh[j]) / (-dir_y);
        } else { // dir_y > 0
            if (std::abs(yp - yh[j+1]) < epsilon) {
                // Already at the upper boundary, move to the next cell
                j = (j + 1) % n_y;
                yp = yh[j] + epsilon;
                continue;
            }
            ly = (yh[j+1] - yp) / dir_y;
        }
        
        Float lx;
        if (std::abs(dir_x) < epsilon) {
            lx = 100000.;
        } else if (dir_x < 0) {
            if (std::abs(xp - xh[i]) < epsilon) {
                // Already at the lower boundary, move to the previous cell
                i = (i == 0) ? n_x - 1 : i - 1;
                xp = xh[i+1] - epsilon;
                continue;
            }
            lx = (xp - xh[i]) / (-dir_x);
        } else { // dir_x > 0
            if (std::abs(xp - xh[i+1]) < epsilon) {
                // Already at the upper boundary, move to the next cell
                i = (i + 1) % n_x;
                xp = xh[i] + epsilon;
                continue;
            }
            lx = (xh[i+1] - xp) / dir_x;
        }

        Float l = std::min({lx, ly, lz});
        Float dx0 = l * dir_x;
        Float dy0 = l * dir_y;
        Float dz0 = l * dir_z;
        // Move along axes:
        xp += dx0;
        yp += dy0;
        zp += dz0;
        
        // Record the path segment
        dz_tilted[z_idx] += dz0;
        
        // Check z boundary crossing
        if (std::abs(l - lz) < epsilon || zp >= zh[k+1]) {
            k += 1;
            // Create a new path segment after crossing boundary
            tilted_path.push_back({i, j, k});
            dz_tilted.push_back(0.0);
            z_idx += 1;
        }
        
        // Check y boundary crossing
        if (std::abs(l - ly) < epsilon) {

            j = int(j + sign(dy0));
            j = (j == -1) ? n_y - 1 : j%n_y;
            yp = dy0 < 0 ? yh[j+1] : yh[j];
            
        }
        
        // Check x boundary crossing
        if (std::abs(l - lx) < epsilon) {
            i = int(i + sign(dx0));
            i = (i == -1) ? n_x - 1 : i%n_x;
            xp = dx0 < 0 ? xh[i+1] : xh[i];
        }
    }    
    // Construct final zh_tilted
    zh_tilted.clear();
    zh_tilted.push_back(0.);
    for (int iz = 0; iz < dz_tilted.size(); ++iz) {
        zh_tilted.push_back(zh_tilted[iz] + dz_tilted[iz]);
    }
}


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
    bool switch_liq_cloud_optics, bool switch_ice_cloud_optics
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
                         Array<Float,2>* lwp_out,
                         Array<Float,2>* iwp_out,
                         Array<Float,2>* rel_out,
                         Array<Float,2>* dei_out,
                         const bool switch_liq_cloud_optics,
                         const bool switch_ice_cloud_optics)
{
    const int total_cols = n_col_x * n_col_y;
    for (int idx_x = 0; idx_x < n_col_x; ++idx_x) {
        for (int idx_y = 0; idx_y < n_col_y; ++idx_y) {
            int col_idx = idx_x + idx_y * n_col_x;
            const ColumnResult& col = col_results[col_idx];
            for (int j = 0; j < n_z; ++j) {
                int out_idx = col_idx + j * total_cols;
                if (switch_liq_cloud_optics) {
                    lwp_out->v()[out_idx] = col.lwp.v()[j];
                    rel_out->v()[out_idx] = col.rel.v()[j];
                }
                if (switch_ice_cloud_optics) {
                    iwp_out->v()[out_idx] = col.iwp.v()[j];
                    dei_out->v()[out_idx] = col.dei.v()[j];
                }
            }
        }
    }
}

void compress_columns_weighted_avg(const int n_x, const int n_y,  
                      const int n_out, 
                      const int n_tilt,
                      const int compress_lay_start_idx,
                      std::vector<Float>& var, std::vector<Float>& var_weighting)
{
    std::vector<Float> var_tmp(n_out * n_x * n_y);

    #pragma omp parallel for
    for (int ilay = 0; ilay < compress_lay_start_idx; ++ilay)
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
    for (int ilay = compress_lay_start_idx; ilay < n_out; ++ilay)
    {
        const int in_offset = ilay - compress_lay_start_idx;
        const int i_lay_in = (compress_lay_start_idx + 2 * in_offset);
        int num_inputs;
        if (ilay < (n_out - 1)) {
            num_inputs = 2;
        } else {
            num_inputs = ((i_lay_in + 1) == (n_tilt - 1)) ? 2 : 3;
        }

        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                Float t_sum = 0.0;
                Float w_sum = 0.0;
                for (int k = 0; k < num_inputs; ++k)
                {
                    int in_idx = ix + iy * n_x + (i_lay_in + k) * n_x * n_y;
                    t_sum += var[in_idx] * var_weighting[in_idx];
                    w_sum += var_weighting[in_idx];
                }

                if (w_sum > 1e-6)
                {
                    var_tmp[out_idx] = t_sum / w_sum;
                } 
                else 
                {
                    Float avg = 0.0;
                    for (int k = 0; k < num_inputs; ++k)
                    {
                        int in_idx = ix + iy * n_x + (i_lay_in + k) * n_x * n_y;
                        avg += var[in_idx];
                    }
                    var_tmp[out_idx] = avg / num_inputs;
                }
            }
        }
    }
    var = var_tmp;
}

void compress_columns_p_or_t(const int n_x, const int n_y, 
                      const int n_out_lay,  const int n_tilt,
                      const int compress_lay_start_idx,
                      std::vector<Float>& var_lev, std::vector<Float>& var_lay)
{
    std::vector<Float> var_tmp_lay(n_out_lay * n_x * n_y);
    std::vector<Float> var_tmp_lev((n_out_lay + 1) * n_x * n_y);

    #pragma omp parallel for
    for (int ilay = 0; ilay < compress_lay_start_idx; ++ilay)
    {
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                var_tmp_lay[out_idx] = var_lev[out_idx];
                var_tmp_lev[out_idx] = var_lev[out_idx];
            }
        }
    }

    int ilay = compress_lay_start_idx;
    for (int iy = 0; iy < n_y; ++iy)
    {
        for (int ix = 0; ix < n_x; ++ix)
        {
            const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
            var_tmp_lev[out_idx] = var_lev[out_idx];
        }
    }

    #pragma omp parallel for
    for (int ilev = (compress_lay_start_idx + 1); ilev < (n_out_lay + 1); ++ilev)
    {
        int i_lev_in;
        if (ilev == n_out_lay)
        {
            i_lev_in = n_tilt;
        }
        else
        {
            i_lev_in = (compress_lay_start_idx + 2) + 2 * (ilev - (compress_lay_start_idx + 1));
        }

        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilev * n_x * n_y;
                const int in_idx = ix + iy * n_x + i_lev_in * n_x * n_y;
                var_tmp_lev[out_idx] = var_lev[in_idx];
            }
        }
    }

    #pragma omp parallel for
    for (int ilay = compress_lay_start_idx; ilay < n_out_lay; ++ilay)
    {
        const int in_offset = ilay - compress_lay_start_idx;
        int i_lev_to_lay_in;
        if (ilay == (n_out_lay - 1))
        {
            i_lev_to_lay_in = n_tilt - 1; // in some cases this is a slight approximation.
        }
        else 
        {
            i_lev_to_lay_in = (compress_lay_start_idx + 2 * in_offset - 1);
        }
        
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                const int in_idx_lev_to_lay = ix + iy * n_x + i_lev_to_lay_in * n_x * n_y;
                
                var_tmp_lay[out_idx] = var_lev[in_idx_lev_to_lay];
            }
        }
    }
    var_lev = var_tmp_lev;
    var_lay = var_tmp_lay;
}

void tilt_fields(const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1> zh, const Array<Float,1> z,
    const Array<Float,1> zh_tilt, const Array<ijk,1> path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Gas_concs& gas_concs_copy, const std::vector<std::string> gas_names
) {
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

void compress_fields(const int compress_lay_start_idx, const int n_col_x, const int n_col_y,
    const int n_z_in, const int n_zh_in,  const int n_z_tilt,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Gas_concs& gas_concs_copy, std::vector<std::string> gas_names)
{
    const int n_col = n_col_x*n_col_y;

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

void create_tilted_columns(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                           const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                           std::vector<Float>& var)
{
    const int n_lay = tilted_path.size();
    const int n_lev = zh_tilted.size();

    std::vector<Float> var_tmp(n_lay*n_x*n_y);

    #pragma omp parallel for
    for (int ilay=0; ilay<n_lay; ++ilay)
    {
        const ijk offset = tilted_path[ilay];
        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y + ilay*n_y*n_x;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + offset.k*n_y*n_x;
                var_tmp[idx_out] = var[idx_in];
            } 
    }

    var.resize(var_tmp.size());
    var = var_tmp;
}

void interpolate(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                 const std::vector<Float>& zh_in, const std::vector<Float>& zf_in,
                 const std::vector<Float>& play_in, const std::vector<Float>& plev_in,
                 const Float zp, const ijk offset,
                 Float* p_out)
{
    int zp_in_zh = -1; // half level
    int zp_in_zf = -1; // full level
    for (int ilev=0; ilev<n_lev_in; ++ilev)
        if (std::abs(zh_in[ilev]-zp) < 1e-2)
            zp_in_zh= ilev;
    for (int ilay=0; ilay<n_lay_in; ++ilay)
        if (std::abs(zf_in[ilay]-zp) < 1e-2)
            zp_in_zf = ilay;
    if (zp_in_zh > -1)
    {
        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + zp_in_zh*n_y*n_x;
                p_out[idx_out] = plev_in[idx_in];
            } 
    }
    else if (zp_in_zf > -1)
    {
        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + zp_in_zf*n_y*n_x;
                p_out[idx_out] = play_in[idx_in];
            } 
    }
    else
    {
        int posh_bot = 0;
        int posf_bot = 0;
        for (int ilev=0; ilev<n_lev_in-1; ++ilev)
            if (zh_in[ilev] < zp)
                posh_bot = ilev;     
        for (int ilay=0; ilay<n_lay_in; ++ilay)
            if (zf_in[ilay] < zp)
                posf_bot = ilay;         
        const Float* p_top;
        const Float* p_bot;
        Float  z_top;
        Float  z_bot;

        const int zh_top = zh_in[posh_bot+1];
        const int zh_bot = zh_in[posh_bot];
        const int zf_top = (posf_bot+1 < n_lay_in) ? zf_in[posf_bot+1] : zh_top+1;
        const int zf_bot = zf_in[posf_bot];

        if (zh_top > zf_top)
        {
            p_top = &play_in.data()[(posf_bot+1)*n_x*n_y];       
            z_top = zf_in[posf_bot+1];       
        }
        else
        {   
            p_top = &plev_in.data()[(posh_bot+1)*n_x*n_y];       
            z_top = zh_in[posh_bot+1];
        }
        if (zh_bot < zf_bot)
        {
            p_bot = &play_in.data()[(posf_bot)*n_x*n_y];       
            z_bot = zf_in[posf_bot];       
        }
        else
        {   
            p_bot = &plev_in.data()[(posh_bot)*n_x*n_y];       
            z_bot = zh_in[posh_bot];
        }

        Float dz = z_top-z_bot;

        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x;
                const Float pres = (zp-z_bot)/dz*p_top[idx_in] + (z_top-zp)/dz*p_bot[idx_in];
                p_out[idx_out] = pres;
            } 

    }
}


void create_tilted_columns_levlay(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                                 const std::vector<Float>& zh_in, const std::vector<Float>& z_in,
                                 const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                                 std::vector<Float>& var_lay, std::vector<Float>& var_lev)

{
    const int n_lay = tilted_path.size();
    const int n_lev = zh_tilted.size();
    std::vector<Float> z_tilted(n_lay);
    for (int ilay=0; ilay<n_lay; ++ilay)
        z_tilted[ilay] = (zh_tilted[ilay]+zh_tilted[ilay+1])/Float(2.);

    std::vector<Float> var_lay_tmp(n_lay*n_x*n_y);
    std::vector<Float> var_lev_tmp(n_lev*n_x*n_y);
    for (int iy=0; iy<n_y; ++iy)
        for (int ix=0; ix<n_x; ++ix)
        {
            const int idx = ix + iy*n_y;
            var_lev_tmp[idx] = var_lev[idx];
        } 

    #pragma omp parallel for
    for (int ilev=1; ilev<n_lev; ++ilev)
        interpolate(n_x, n_y, n_lay_in, n_lev_in, zh_in, z_in, var_lay, var_lev, zh_tilted[ilev], tilted_path[ilev-1], &var_lev_tmp.data()[ilev*n_y*n_x]);
    #pragma omp parallel for
    for (int ilay=0; ilay<n_lay; ++ilay)
        interpolate(n_x, n_y, n_lay_in, n_lev_in, zh_in, z_in, var_lay, var_lev, z_tilted[ilay], tilted_path[ilay], &var_lay_tmp.data()[ilay*n_y*n_x]);

    var_lay.resize(n_lay*n_y*n_x);
    var_lev.resize(n_lev*n_y*n_x);
    var_lay = var_lay_tmp;
    var_lev = var_lev_tmp;
}


void tica_tilt(
    const Float sza, const Float azi,
    const int n_col_x, const int n_col_y, const int n_col,
    const int n_lay, const int n_lev, const int n_z_in, const int n_zh_in ,
    Array<Float,1> xh, Array<Float,1> yh, Array<Float,1> zh, Array<Float,1> z,
    Array<Float,2> p_lay, Array<Float,2> t_lay, Array<Float,2> p_lev, Array<Float,2> t_lev, 
    Array<Float,2> lwp, Array<Float,2> iwp, Array<Float,2> rel, Array<Float,2> dei, 
    Gas_concs gas_concs,
    Array<Float,2>& p_lay_out, Array<Float,2>& t_lay_out, Array<Float,2>& p_lev_out, Array<Float,2>& t_lev_out, 
    Array<Float,2>& lwp_out, Array<Float,2>& iwp_out, Array<Float,2>& rel_out, Array<Float,2>& dei_out, 
    Gas_concs& gas_concs_out, 
    std::vector<std::string> gas_names,
    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics
)
{
       // if t lev all 0, interpolate from t lay
       if (*std::max_element(t_lev_out.v().begin(), t_lev_out.v().end()) <= 0) {
       for (int i = 1; i <= n_col; ++i) {
           for (int j = 2; j <= n_lay; ++j) {
               t_lev_out({i, j}) = (t_lay_out({i, j}) + t_lay_out({i, j - 1})) / 2.0;
           }
           t_lev_out({i, n_lev}) = 2 * t_lay_out({i, n_lay}) - t_lev_out({i,n_lay});
           t_lev_out({i, 1}) = 2 * t_lay_out({i, 1}) - t_lev_out({i,2});
           }
       }
       // copy interpolated values into t_lev too
       t_lev = t_lev_out;

       ////// SETUP FOR CENTER START POINT TILTING //////
       Array<ijk,1> center_path;
       Array<Float,1> center_zh_tilt;
       tilted_path(xh.v(),yh.v(),zh.v(),z.v(),sza,azi, 0.5, 0.5, center_path.v(), center_zh_tilt.v());
       
       int n_zh_tilt_center = center_zh_tilt.v().size();
       int n_z_tilt_center = n_zh_tilt_center - 1;
   
       tilt_fields(n_z_in, n_zh_in, n_col_x, n_col_y,
           n_z_tilt_center, n_zh_tilt_center, n_col,
           zh, z,
           center_zh_tilt, center_path,
           &p_lay_out, &t_lay_out, &p_lev_out, &t_lev_out, 
           gas_concs_out, gas_names
       );
       
       int idx_hold = 2*(n_z_tilt_center - n_z_in);
       if ((n_z_tilt_center - idx_hold) % 2 != 0) {
           idx_hold--;
       }
   
       int compress_lay_start_idx_center = (n_z_tilt_center - idx_hold);
       if (compress_lay_start_idx_center < 0) {
           throw std::runtime_error("compress_lay_start_idx is negative - SZA too high.");
       }
       compress_fields(compress_lay_start_idx_center, n_col_x, n_col_y,
                   n_z_in, n_zh_in, n_z_tilt_center,
                   &p_lay_out, &t_lay_out, &p_lev_out, &t_lev_out, 
                   gas_concs_out, gas_names);
       
       ////// SETUP FOR RANDOM START POINT TILTING //////    
       if (switch_cloud_optics)
       {
           std::vector<std::pair<Float, Float>> x_y_start_arr(n_col);
           std::vector<Array<ijk,1>> by_col_paths(n_col);
           std::vector<Array<Float,1>> by_col_zh_tilt(n_col);
           std::vector<Float> by_col_n_zh_tilt(n_col);
           std::vector<Float> by_col_n_z_tilt(n_col);
       
           std::mt19937_64 rng;
           uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
           std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
           rng.seed(ss);
           std::uniform_real_distribution<double> unif(0.001, 0.999);
           for (int i = 0; i < n_col; i++)
           {
               Float x_point = unif(rng);
               Float y_point = unif(rng);
               x_y_start_arr[i] = std::make_pair(x_point, y_point);
           }
       
           Status::print_message("###### Starting Tilting ######");
           auto time_start = std::chrono::high_resolution_clock::now();
           #pragma omp parallel for
           for (int i = 0; i < n_col; i++) {
               Float x_start = x_y_start_arr[i].first;
               Float y_start = x_y_start_arr[i].second;
               Array<ijk,1> path;
               Array<Float,1> zh_tilt;
               
               tilted_path(xh.v(), yh.v(), zh.v(), z.v(), sza, azi, x_start, y_start, path.v(), zh_tilt.v());
               int n_zh_tilt = zh_tilt.v().size();
               int n_z_tilt = n_zh_tilt - 1;

               path.set_dims({n_z_tilt}); 
               zh_tilt.set_dims({n_zh_tilt}); 
       
               by_col_paths[i] = path;
               by_col_zh_tilt[i] = zh_tilt;
               by_col_n_zh_tilt[i] = n_zh_tilt;
               by_col_n_z_tilt[i] = n_zh_tilt - 1;
           }
           auto time_end = std::chrono::high_resolution_clock::now();
           auto duration = std::chrono::duration<double, std::milli>(time_end-time_start).count();
       
           Status::print_message("Duration tilting columns: " + std::to_string(duration) + " (ms)");
           Status::print_message("###### Finish Tilting ######");
       
           Status::print_message("###### Check Sizes ######");
           std::vector<Float> by_col_compress_start(n_col);
           int max_n_z_tilt = 0;
           for (int i = 0; i < n_col; i++) {
               const Float n_zh_tilt = by_col_n_zh_tilt[i];
               const Float n_z_tilt = by_col_n_z_tilt[i];
               if (n_z_tilt > max_n_z_tilt) {
                   max_n_z_tilt = n_z_tilt;
               }
               int idx_hold = 2*(n_z_tilt - n_z_in);
       
               int n_lay_compress = (n_z_tilt - idx_hold) + (idx_hold)/2;
               int n_lev_compress = n_lay_compress + 1;
               int compress_lay_start_idx = (n_z_tilt - idx_hold);
               if (compress_lay_start_idx < 0) {
                   throw std::runtime_error("compress_lay_start_idx is negative - SZA too high.");
               }
               by_col_compress_start[i] = compress_lay_start_idx;
           }
   
           const int total_iterations = n_col_x * n_col_y;
           std::vector<ColumnResult> col_results(total_iterations);
       
           // Do Normalization and Reshaping OUTSIDE OF LOOP
           Array<Float,2> lwp_norm_reshaped;
           lwp_norm_reshaped.set_dims({n_z_in, n_col});
           Array<Float,2> rel_reshaped;
           rel_reshaped.set_dims({n_z_in, n_col});
           Array<Float,2> iwp_norm_reshaped;
           iwp_norm_reshaped.set_dims({n_z_in, n_col});
           Array<Float,2> dei_reshaped;
           dei_reshaped.set_dims({n_z_in, n_col});
   
           for (int icol = 1; icol <= n_col; ++icol)    
           {
               for (int ilay = 1; ilay <= n_zh_in; ++ilay)    
               {
                   Float dz = zh({ilay + 1}) - zh({ilay});
                   if (switch_liq_cloud_optics)
                   {
                       (lwp_norm_reshaped)({ilay, icol}) = (lwp)({icol, ilay})/dz;
                       (rel_reshaped)({ilay, icol}) = (rel)({icol, ilay});
                   }
                   if (switch_ice_cloud_optics)
                   {
                       (iwp_norm_reshaped)({ilay, icol}) = (iwp)({icol, ilay})/dz;
                       (dei_reshaped)({ilay, icol}) = (dei)({icol, ilay});
                   }
               }
           }
           
           if (switch_liq_cloud_optics){
               Array<Float,1> lwp_compress;
               lwp_compress.set_dims({n_z_in});
               Array<Float,1> rel_compress;
               rel_compress.set_dims({n_z_in});
               const std::vector<Float> var_lwp = lwp_norm_reshaped.v();
               const std::vector<Float> var_rel = rel_reshaped.v();
   
               Status::print_message("###### Start Loop ######");
               auto time_start_loop = std::chrono::high_resolution_clock::now();
               for (int idx_y = 0; idx_y < n_col_y; idx_y++)
               {
                   for (int idx_x = 0; idx_x < n_col_x; idx_x++)
                   {
                       int i = idx_x + idx_y * n_col_x;
                       const Array<ijk,1> tilted_path = by_col_paths[i];
                       const std::vector<ijk>& tilted_path_v = tilted_path.v();
                       const Array<Float,1> zh_tilt = by_col_zh_tilt[i];
                       const int n_z_tilt = by_col_n_z_tilt[i];
                       const int compress_lay_start_idx = by_col_compress_start[i];
   
                       std::vector<Float> var_lwp_tmp(n_z_tilt);
                       std::vector<Float> var_rel_tmp(n_z_tilt);
                       std::vector<Float> var_lwp_out(n_z_in);
                       std::vector<Float> var_rel_out(n_z_in);
                       
                       for (int ilay=0; ilay < n_z_tilt; ++ilay)
                       {
                           Float dz = zh_tilt({ilay + 1}) - zh_tilt({ilay});
                           const ijk offset = tilted_path_v[ilay];
                           const int i_col_new  = ((idx_y+offset.j)%n_col_y) * n_col_x + ((idx_x + offset.i)%n_col_x);
                           const int idx_in = offset.k + i_col_new*n_z_in;
                           var_lwp_tmp[ilay] = var_lwp[idx_in] * dz;
                           var_rel_tmp[ilay] = var_rel[idx_in];
                       }  
                       
                       for (int ilay = 0; ilay < compress_lay_start_idx; ++ilay)
                       {
                           var_lwp_out[ilay] = var_lwp_tmp[ilay];
                           var_rel_out[ilay] = var_rel_tmp[ilay];
                       }
               
                       for (int ilay = compress_lay_start_idx; ilay < n_z_in; ++ilay)
                       {
                           const int in_offset = ilay - compress_lay_start_idx;
                           const int i_lay_in = compress_lay_start_idx + 2 * in_offset;
                           int num_inputs;
                           if (ilay < (n_z_in - 1)) {
                               num_inputs = 2;
                           } 
                           else {
                               num_inputs = ((i_lay_in + 1) == (n_z_tilt - 1)) ? 2 : 3;
                           }
                           Float sum = 0.0;
                           Float t_sum = 0.0;
                           Float w_sum = 0.0;
               
                           for (int k = 0; k < num_inputs; ++k)
                           {
                               int in_idx = i_lay_in + k;
                               sum += var_lwp_tmp[in_idx];
                               t_sum += var_rel_tmp[in_idx] * var_lwp_tmp[in_idx];
                               w_sum += var_lwp_tmp[in_idx];
                           }
                           var_lwp_out[ilay] = sum;
                           if (w_sum > 1e-6)
                           {
                               var_rel_out[ilay] = t_sum / w_sum;
                           } 
                           else 
                           {
                               Float avg = 0.0;
                               for (int k = 0; k < num_inputs; ++k)
                               {
                                   int in_idx = (i_lay_in + k);
                                   avg += var_rel_out[in_idx];
                               }
                               var_rel_out[ilay] = avg / num_inputs;
                           }
                       }
   
                       col_results[i].lwp   = std::move(var_lwp_out);
                       col_results[i].rel   = std::move(var_rel_out);
                   }
               }
           
               auto time_end_loop = std::chrono::high_resolution_clock::now();
               auto duration_loop = std::chrono::duration<double, std::milli>(time_end_loop - time_start_loop).count();
               Status::print_message("Duration liq loop: " + std::to_string(duration_loop) + " (ms)");
           }
   
           if (switch_ice_cloud_optics){
               Array<Float,1> iwp_compress;
               iwp_compress.set_dims({n_z_in});
               Array<Float,1> dei_compress;
               dei_compress.set_dims({n_z_in});
               const std::vector<Float> var_iwp = iwp_norm_reshaped.v();
               const std::vector<Float> var_dei = dei_reshaped.v();
           
               Status::print_message("###### Start Loop ######");
               auto time_start_loop = std::chrono::high_resolution_clock::now();
               for (int idx_y = 0; idx_y < n_col_y; idx_y++)
               {
                   for (int idx_x = 0; idx_x < n_col_x; idx_x++)
                   {
                       int i = idx_x + idx_y * n_col_x;
                       const Array<ijk,1> tilted_path = by_col_paths[i];
                       const std::vector<ijk>& tilted_path_v = tilted_path.v();
                       const Array<Float,1> zh_tilt = by_col_zh_tilt[i];
                       const int n_z_tilt = by_col_n_z_tilt[i];
                       const int compress_lay_start_idx = by_col_compress_start[i];
           
                       std::vector<Float> var_iwp_tmp(n_z_tilt);
                       std::vector<Float> var_dei_tmp(n_z_tilt);
                       std::vector<Float> var_iwp_out(n_z_in);
                       std::vector<Float> var_dei_out(n_z_in);
                       
                       for (int ilay=0; ilay < n_z_tilt; ++ilay)
                       {
                           Float dz = zh_tilt({ilay + 1}) - zh_tilt({ilay});
                           const ijk offset = tilted_path_v[ilay];
                           const int i_col_new  = ((idx_y+offset.j)%n_col_y) * n_col_x + ((idx_x + offset.i)%n_col_x);
                           const int idx_in = offset.k + i_col_new*n_z_in;
                           var_iwp_tmp[ilay] = var_iwp[idx_in] * dz;
                           var_dei_tmp[ilay] = var_dei[idx_in];
                       }  
                       
                       for (int ilay = 0; ilay < compress_lay_start_idx; ++ilay)
                       {
                           var_iwp_out[ilay] = var_iwp_tmp[ilay];
                           var_dei_out[ilay] = var_dei_tmp[ilay];
                       }
               
                       for (int ilay = compress_lay_start_idx; ilay < n_z_in; ++ilay)
                       {
                           const int in_offset = ilay - compress_lay_start_idx;
                           const int i_lay_in = compress_lay_start_idx + 2 * in_offset;
                           int num_inputs;
                           if (ilay < (n_z_in - 1)) {
                               num_inputs = 2;
                           } 
                           else {
                               num_inputs = ((i_lay_in + 1) == (n_z_tilt - 1)) ? 2 : 3;
                           }
                           Float sum = 0.0;
                           Float t_sum = 0.0;
                           Float w_sum = 0.0;
               
                           for (int k = 0; k < num_inputs; ++k)
                           {
                               int in_idx = i_lay_in + k;
                               sum += var_iwp_tmp[in_idx];
                               t_sum += var_dei_tmp[in_idx] * var_iwp_tmp[in_idx];
                               w_sum += var_iwp_tmp[in_idx];
                           }
                           var_iwp_out[ilay] = sum;
                           if (w_sum > 1e-6)
                           {
                               var_dei_out[ilay] = t_sum / w_sum;
                           } 
                           else 
                           {
                               Float avg = 0.0;
                               for (int k = 0; k < num_inputs; ++k)
                               {
                                   int in_idx = (i_lay_in + k);
                                   avg += var_dei_out[in_idx];
                               }
                               var_dei_out[ilay] = avg / num_inputs;
                           }
                       }
           
                       col_results[i].iwp   = std::move(var_iwp_out);
                       col_results[i].dei   = std::move(var_dei_out);
                   }
               }
           
               auto time_end_loop = std::chrono::high_resolution_clock::now();
               auto duration_loop = std::chrono::duration<double, std::milli>(time_end_loop - time_start_loop).count();
               Status::print_message("Duration ice loop: " + std::to_string(duration_loop) + " (ms)");
           }
           
           post_process_output(col_results, n_col_x, n_col_y, n_z_in, n_zh_in,
               &lwp_out, &iwp_out, &rel_out, &dei_out,
               switch_liq_cloud_optics, switch_ice_cloud_optics);
   
       }
   
       restore_bkg_profile_bundle(n_col_x, n_col_y, 
           n_lay, n_lev, n_lay, n_lev, 
           n_z_in, n_zh_in, n_z_in, n_zh_in,
           &p_lay_out, &t_lay_out, &p_lev_out, &t_lev_out, 
           &lwp_out, &iwp_out, &rel_out, &dei_out,
           gas_concs_out,
           &p_lay, &t_lay, &p_lev, &t_lev, 
           &lwp, &iwp, &rel, &dei,
           gas_concs, 
           gas_names,
           switch_liq_cloud_optics, switch_ice_cloud_optics
       );
}


