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
#include <random>
#include <cmath>
#include <iomanip>
#include <cuda_profiler_api.h>

#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "Gas_concs.h"
#include "tilted_column.h"
#include "compress_column.h"
#include "tilt_prep_utils.h"
#include "types.h"


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
    ////// FINISH READING THE ATMOSPHERIC DATA //////

    // if t lev all 0, interpolate from t lay
    if (*std::max_element(t_lev.v().begin(), t_lev.v().end()) <= 0) {
        for (int i = 1; i <= n_col; ++i) {
            for (int j = 2; j <= n_lay; ++j) {
                t_lev({i, j}) = (t_lay({i, j}) + t_lay({i, j - 1})) / 2.0;
            }
            t_lev({i, n_lev}) = 2 * t_lay({i, n_lay}) - t_lev({i,n_lay});
            t_lev({i, 1}) = 2 * t_lay({i, 1}) - t_lev({i,2});
        }
    }

    ////// SETUP FOR TILTING //////

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
    for (int i = 0; i < n_col; i++) {
        const Float n_zh_tilt = by_col_n_zh_tilt[i];
        const Float n_z_tilt = by_col_n_z_tilt[i];
        int idx_hold = 2*(n_z_tilt - n_z_in);

        int n_lay_compress = (n_z_tilt - idx_hold) + (idx_hold)/2;
        int n_lev_compress = n_lay_compress + 1;
        int compress_lay_start_idx = (n_z_tilt - idx_hold);
        if (compress_lay_start_idx < 0) {
            throw std::runtime_error("compress_lay_start_idx is negative - SZA too high.");
        }
    }

    Status::print_message("###### Make output arrays ######");
    Array<Float,2> lwp_out;
    lwp_out.set_dims({n_col, n_z_in});
    Array<Float,2> rel_out;
    rel_out.set_dims({n_col, n_z_in});

    Array<Float,2> iwp_out;
    iwp_out.set_dims({n_col, n_z_in});
    Array<Float,2> dei_out;
    dei_out.set_dims({n_col, n_z_in});

    Array<Float,2> p_lay_out;
    p_lay_out.set_dims({n_col, n_z_in});

    Array<Float,2> t_lay_out;
    t_lay_out.set_dims({n_col, n_z_in});

    Array<Float,2> p_lev_out;
    p_lev_out.set_dims({n_col, n_zh_in});

    Array<Float,2> t_lev_out;
    t_lev_out.set_dims({n_col, n_zh_in});

    Gas_concs gas_concs_out = gas_concs;
    for (const auto& gas_name : gas_names) {
        if (!gas_concs_out.exists(gas_name)) {
            continue;
        }
        const Array<Float,2>& gas = gas_concs_out.get_vmr(gas_name);
        if (gas.size() > 1) {
            Array<Float,2> gas_tmp;
            gas_tmp.set_dims({n_col, n_z_in});
            gas_concs_out.set_vmr(gas_name, gas_tmp);
        }
    }

    const int total_iterations = n_col_x * n_col_y;
    double total_tilt_time = 0.0;
    double total_compress_time = 0.0;
    double total_add_time = 0.0;

    std::vector<ColumnResult> col_results(total_iterations);

    Status::print_message("###### Start Loop ######");
    std::cout << "n_col: " << n_col << std::endl;

    auto time_start_loop = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int idx_x = 0; idx_x < n_col_x; idx_x++)
    {
        for (int idx_y = 0; idx_y < n_col_y; idx_y++)
        {
            int i = idx_x * n_col_y + idx_y;

            Array<Float,2> lwp_copy = lwp;
            Array<Float,2> iwp_copy = iwp;
            Array<Float,2> rel_copy = rel;
            Array<Float,2> dei_copy = dei;
            Array<Float,2> t_lay_copy = t_lay;
            Array<Float,2> t_lev_copy = t_lev;
            Array<Float,2> p_lay_copy = p_lay;
            Array<Float,2> p_lev_copy = p_lev;
            Gas_concs gas_concs_copy = gas_concs;

            const Array<ijk,1> path = by_col_paths[i];
            const Array<Float,1> zh_tilt = by_col_zh_tilt[i];
            const Float n_zh_tilt = by_col_n_zh_tilt[i];
            const Float n_z_tilt = by_col_n_z_tilt[i];

            tilt_fields_single_column(idx_x, idx_y, n_z_in, n_zh_in, n_col_x, n_col_y,
                n_z_tilt, n_zh_tilt, n_col,
                zh, z,
                zh_tilt, path,
                &p_lay_copy, &t_lay_copy, &p_lev_copy, &t_lev_copy, 
                &lwp_copy, &iwp_copy, &rel_copy, &dei_copy, 
                gas_concs_copy, gas_names,
                switch_cloud_optics, switch_liq_cloud_optics, switch_ice_cloud_optics
            );

            std::vector<Float> z_out = linspace(z.v()[0], z.v()[n_z_in - 1], n_z_tilt);
            std::vector<Float> zh_out = linspace(zh.v()[0], zh.v()[n_zh_in - 1], n_zh_tilt);

            int idx_hold = 2 * (n_z_tilt - n_z_in);
            if ((z_out.size() - idx_hold) % 2 != 0) {
                idx_hold--;
            }

            int n_lay_compress = (n_z_tilt - idx_hold) + (idx_hold) / 2;
            int n_lev_compress = n_lay_compress + 1;
            int compress_lay_start_idx = (n_z_tilt - idx_hold);
            if (compress_lay_start_idx < 0) {
                throw std::runtime_error("compress_lay_start_idx is negative - SZA too high.");
            }
            std::vector<Float> z_out_compress = linspace(z.v()[0], z.v()[n_z_in - 1], n_lay_compress);
            std::vector<Float> zh_out_compress = linspace(zh.v()[0], zh.v()[n_zh_in - 1], n_lev_compress);
            assert(n_lev_compress == n_zh_in && n_lay_compress == n_z_in);

            compress_fields(compress_lay_start_idx, 1, 1,
                n_z_in, n_zh_in, n_z_tilt,
                &p_lay_copy, &t_lay_copy, &p_lev_copy, &t_lev_copy, 
                &lwp_copy, &iwp_copy, &rel_copy, &dei_copy, 
                gas_concs_copy, gas_names,
                switch_cloud_optics, switch_liq_cloud_optics, switch_ice_cloud_optics);

            col_results[i].p_lay = std::move(p_lay_copy);
            col_results[i].t_lay = std::move(t_lay_copy);
            col_results[i].p_lev = std::move(p_lev_copy);
            col_results[i].t_lev = std::move(t_lev_copy);
            col_results[i].lwp   = std::move(lwp_copy);
            col_results[i].iwp   = std::move(iwp_copy);
            col_results[i].rel   = std::move(rel_copy);
            col_results[i].dei   = std::move(dei_copy);
            col_results[i].gas_concs = std::move(gas_concs_copy);
        }
    }

    auto time_end_loop = std::chrono::high_resolution_clock::now();
    auto duration_loop = std::chrono::duration<double, std::milli>(time_end_loop - time_start_loop).count();
    Status::print_message("Duration loop: " + std::to_string(duration_loop) + " (ms)");

    std::vector<Float> z_out_compress = linspace(z.v()[0], z.v()[n_z_in - 1], n_z_in);
    std::vector<Float> zh_out_compress = linspace(zh.v()[0], zh.v()[n_zh_in - 1], n_zh_in);

    post_process_output(col_results, n_col_x, n_col_y, n_z_in, n_zh_in,
                    &p_lay_out, &t_lay_out, &p_lev_out, &t_lev_out, 
                    &lwp_out, &iwp_out, &rel_out, &dei_out,
                    gas_concs_out,
                    gas_names, 
                    switch_cloud_optics, switch_liq_cloud_optics, switch_ice_cloud_optics);

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
        switch_cloud_optics, switch_liq_cloud_optics, switch_ice_cloud_optics
    );

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
