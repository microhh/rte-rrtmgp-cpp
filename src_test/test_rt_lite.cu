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
#include <cuda_profiler_api.h>


#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "Raytracer.h"
#include "raytracer_kernels.h"
#include "types.h"
#include "tools_gpu.h"


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

void solve_radiation(int argc, char** argv)
{
    Status::print_message("###### Starting RTE+RRTMGP solver ######");

    ////// FLOW CONTROL SWITCHES //////
    // Parse the command line options.
    std::map<std::string, std::pair<bool, std::string>> command_line_switches {
        {"raytracing"        , { true,  "Use raytracing for flux computation. '--raytracing 256': use 256 rays per pixel" }},
        {"independent-column", { false, "run raytracer in independent column mode"}},
        {"profiling"         , { false, "Perform additional profiling run."         }} };

    std::map<std::string, std::pair<int, std::string>> command_line_ints {
        {"raytracing", {32, "Number of rays initialised at TOD per pixel per quadraute."}}} ;

    if (parse_command_line_options(command_line_switches, command_line_ints, argc, argv))
        return;

    const bool switch_raytracing         = command_line_switches.at("raytracing"       ).first;
    const bool switch_independent_column =  command_line_switches.at("independent-column").first;
    const bool switch_profiling          = command_line_switches.at("profiling"        ).first;
    
    // Print the options to the screen.
    print_command_line_options(command_line_switches, command_line_ints);

    Int photons_per_pixel = Int(command_line_ints.at("raytracing").first);
    if (Float(int(std::log2(Float(photons_per_pixel)))) != std::log2(Float(photons_per_pixel)))
    {
        std::string error = "number of photons per pixel should be a power of 2 ";
        throw std::runtime_error(error);
    }

    Status::print_message("Using "+ std::to_string(photons_per_pixel) + " rays per pixel");


    ////// READ THE ATMOSPHERIC DATA //////
    Status::print_message("Reading atmospheric input data from NetCDF.");

    Netcdf_file input_nc("rt_lite_input.nc", Netcdf_mode::Read);
    const int nx = input_nc.get_dimension_size("x");
    const int ny = input_nc.get_dimension_size("y");
    const int nz = input_nc.get_dimension_size("z");
    const int ncol = nx*ny;
    const Vector<int> grid_cells = {nx, ny, nz};
    
    // Read the x,y,z dimensions if raytracing is enabled
    const Array<Float,1> grid_x(input_nc.get_variable<Float>("x", {nx}), {nx});
    const Array<Float,1> grid_y(input_nc.get_variable<Float>("y", {ny}), {ny});
    const Array<Float,1> grid_z(input_nc.get_variable<Float>("z", {nz}), {nz});

    const Float dx = grid_x({2}) - grid_x({1});
    const Float dy = grid_y({2}) - grid_y({1});
    const Float dz = grid_z({2}) - grid_z({1});
    const Vector<Float> grid_d = {dx, dy, dz};

    const int ngrid_x = input_nc.get_variable<Float>("ngrid_x");
    const int ngrid_y = input_nc.get_variable<Float>("ngrid_y");
    const int ngrid_z = input_nc.get_variable<Float>("ngrid_z");
    const Vector<int> kn_grid = {ngrid_x, ngrid_y, ngrid_z};

    // Read the atmospheric fields.
    const Array<Float,2> tot_tau(input_nc.get_variable<Float>("tot_tau", {nz, ny, nx}), {ncol, nz});
    const Array<Float,2> tot_ssa(input_nc.get_variable<Float>("tot_ssa", {nz, ny, nx}), {ncol, nz});

    Array<Float,2> cld_tau({ncol, nz});	
    Array<Float,2> cld_ssa({ncol, nz});	
    Array<Float,2> cld_asy({ncol, nz});
    
    cld_tau = std::move(input_nc.get_variable<Float>("cld_tau", {nz, ny, nx}));
    cld_ssa = std::move(input_nc.get_variable<Float>("cld_ssa", {nz, ny, nx}));
    cld_asy = std::move(input_nc.get_variable<Float>("cld_asy", {nz, ny, nx}));
    
    Array<Float,2> aer_tau({ncol, nz});	
    Array<Float,2> aer_ssa({ncol, nz});	
    Array<Float,2> aer_asy({ncol, nz});
    
    aer_tau = std::move(input_nc.get_variable<Float>("aer_tau", {nz, ny, nx}));
    aer_ssa = std::move(input_nc.get_variable<Float>("aer_ssa", {nz, ny, nx}));
    aer_asy = std::move(input_nc.get_variable<Float>("aer_asy", {nz, ny, nx}));
    
    // read albedo, solar angles, and top-of-domain fluxes
    Array<Float,2> sfc_albedo({1,ncol});
    sfc_albedo.fill(input_nc.get_variable<Float>("albedo"));
    const Float zenith_angle = input_nc.get_variable<Float>("sza");
    const Float azimuth_angle = input_nc.get_variable<Float>("azi");
    const Float tod_dir = input_nc.get_variable<Float>("tod_direct");
    const Float tod_dif = input_nc.get_variable<Float>("tod_diffuse");

    // output arrays
    Array_gpu<Float,2> flux_tod_dn({nx, ny});
    Array_gpu<Float,2> flux_tod_up({nx, ny});
    Array_gpu<Float,2> flux_sfc_dir({nx, ny});
    Array_gpu<Float,2> flux_sfc_dif({nx, ny});
    Array_gpu<Float,2> flux_sfc_up({nx, ny});
    Array_gpu<Float,3> flux_abs_dir({nx, ny, nz});
    Array_gpu<Float,3> flux_abs_dif({nx, ny, nz});

    
    // empty arrays (mie scattering not supported in lite version)
    Array_gpu<Float,2> mie_cdfs_sub;
    Array_gpu<Float,3> mie_angs_sub;
    Array_gpu<Float,2> rel;

    ////// CREATE THE OUTPUT FILE //////
    // Create the general dimensions and arrays.
    Status::print_message("Preparing NetCDF output file.");

    Netcdf_file output_nc("rt_lite_output.nc", Netcdf_mode::Create);
    output_nc.add_dimension("x", nx);
    output_nc.add_dimension("y", ny);
    output_nc.add_dimension("z", nz);

    //// GPU arrays
    Array_gpu<Float,2> tot_tau_g(tot_tau);
    Array_gpu<Float,2> tot_ssa_g(tot_ssa);
    Array_gpu<Float,2> cld_tau_g(cld_tau);
    Array_gpu<Float,2> cld_ssa_g(cld_ssa);
    Array_gpu<Float,2> cld_asy_g(cld_asy);
    Array_gpu<Float,2> aer_tau_g(aer_tau);
    Array_gpu<Float,2> aer_ssa_g(aer_ssa);
    Array_gpu<Float,2> aer_asy_g(aer_asy);
    Array_gpu<Float,2> sfc_albedo_g(sfc_albedo);
    
    Raytracer raytracer;

    // Solve the radiation.
    Status::print_message("Starting the raytracer!!");

    auto run_solver = [&]()
    {
        cudaDeviceSynchronize();
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        // do something.
        
	    raytracer.trace_rays(
               0,
               switch_independent_column,
               photons_per_pixel,
               grid_cells,
               grid_d,
               kn_grid,
               mie_cdfs_sub,
               mie_angs_sub,
               tot_tau_g,
               tot_ssa_g,
               cld_tau_g,
               cld_ssa_g,
               cld_asy_g,
               aer_tau_g,
               aer_ssa_g,
               aer_asy_g,
               rel,
               sfc_albedo_g, 
               zenith_angle,
               azimuth_angle,
               tod_dir,
               tod_dif,
               flux_tod_dn,
               flux_tod_up,
               flux_sfc_dir,
               flux_sfc_dif,
               flux_sfc_up,
               flux_abs_dir,
               flux_abs_dif);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float duration = 0.f;
        cudaEventElapsedTime(&duration, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        Status::print_message("Duration raytracer: " + std::to_string(duration) + " (ms)");
    };

    // Tuning step;
    run_solver();

    //// Profiling step;
    if (switch_profiling)
    {
        cudaProfilerStart();
        run_solver();
        cudaProfilerStop();
    }
    // output arrays to cpu
    Array<Float,2> flux_tod_dn_c(flux_tod_dn);
    Array<Float,2> flux_tod_up_c(flux_tod_up);
    Array<Float,2> flux_sfc_dir_c(flux_sfc_dir);
    Array<Float,2> flux_sfc_dif_c(flux_sfc_dif);
    Array<Float,2> flux_sfc_up_c(flux_sfc_up);
    Array<Float,3> flux_abs_dir_c(flux_abs_dir);
    Array<Float,3> flux_abs_dif_c(flux_abs_dif);
    // Store the output.
    Status::print_message("Storing the raytracer output.");

    auto nc_flux_tod_dn     = output_nc.add_variable<Float>("flux_tod_dn" , {"y", "x"});
    auto nc_flux_tod_up     = output_nc.add_variable<Float>("flux_tod_up" , {"y", "x"});
    auto nc_flux_sfc_dir    = output_nc.add_variable<Float>("flux_sfc_dir", {"y", "x"});
    auto nc_flux_sfc_dif    = output_nc.add_variable<Float>("flux_sfc_dif", {"y", "x"});
    auto nc_flux_sfc_up     = output_nc.add_variable<Float>("flux_sfc_up" , {"y", "x"});
    auto nc_flux_abs_dir    = output_nc.add_variable<Float>("abs_dir"     , {"z", "y", "x"});
    auto nc_flux_abs_dif    = output_nc.add_variable<Float>("abs_dif"     , {"z", "y", "x"});

    nc_flux_tod_dn   .insert(flux_tod_dn_c  .v(), {0, 0});
    nc_flux_tod_up   .insert(flux_tod_up_c  .v(), {0, 0});
    nc_flux_sfc_dir  .insert(flux_sfc_dir_c .v(), {0, 0});
    nc_flux_sfc_dif  .insert(flux_sfc_dif_c .v(), {0, 0});
    nc_flux_sfc_up   .insert(flux_sfc_up_c  .v(), {0, 0});
    nc_flux_abs_dir  .insert(flux_abs_dir_c .v(), {0, 0, 0});
    nc_flux_abs_dif  .insert(flux_abs_dif_c .v(), {0, 0, 0});

    Status::print_message("###### Finished RAYTRACING #####");
}


int main(int argc, char** argv)
{
    try
    {
        solve_radiation(argc, argv);
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
