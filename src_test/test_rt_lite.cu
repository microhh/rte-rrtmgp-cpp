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
#include <iomanip>
#include <cuda_profiler_api.h>

#include "toml.hpp"

#include "status.h"
#include "netcdf_interface.h"
#include "array.h"
#include "raytracer.h"
#include "raytracer_kernels.h"
#include "raytracer_bw.h"
#include "raytracer_kernels_bw.h"
#include "types.h"
#include "rte_solver_kernels_cuda_rt.h"
#include "rte_sw_rt.h"
#include "tools_gpu.h"


template<typename T>
T get_ini_value(const toml::value& ini_file, const std::string& group, const std::string& item)
{
    const T value = toml::find<T>(ini_file, group, item);
    std::cout << "[" << group << "]" << "[" << item << "] = " << value << std::endl;
    return value;
}


template<typename T>
T get_ini_value(const toml::value& ini_file, const std::string& group, const std::string& item, const T default_value)
{
    auto ini_group = toml::find(ini_file, group);
    const T value = toml::find_or(ini_group, item, default_value);
    std::cout << "[" << group << "]" << "[" << item << "] = " << value << std::endl;
    return value;
}


void solve_radiation(int argc, char** argv)
{
    Status::print_message("###### Starting raytracing ######");

    // Read out the case name from the command line parameter.
    if (argc != 2)
    {
        const std::string error = "The raytracer takes exactly one argument, which is the case name";
        throw std::runtime_error(error);
    }
    const std::string case_name(argv[1]);

    const auto settings = toml::parse(case_name + ".ini");

    const bool switch_raytracing         = get_ini_value<bool>(settings, "switches", "raytracing", true);
    const bool switch_bw_raytracing      = get_ini_value<bool>(settings, "switches", "bw-raytracing", true);
    const bool switch_two_stream         = get_ini_value<bool>(settings, "switches", "two-stream", true);
    const bool switch_cloud_mie          = get_ini_value<bool>(settings, "switches", "cloud-mie", false);
    const bool switch_independent_column = get_ini_value<bool>(settings, "switches", "independent-column", false);
    const bool switch_profiling          = get_ini_value<bool>(settings, "switches", "profiling", false);

    Int photons_per_pixel;
    Int photons_per_pixel_bw;
    if (switch_raytracing)
    {
        photons_per_pixel = get_ini_value<Int>(settings, "ints", "raytracing", Int(32));
        if (Float(int(std::log2(Float(photons_per_pixel)))) != std::log2(Float(photons_per_pixel)))
        {
            std::string error = "number of photons per pixel should be a power of 2 ";
            throw std::runtime_error(error);
        }
        Status::print_message("Using "+ std::to_string(photons_per_pixel) + " rays per pixel");
    }

    if (switch_bw_raytracing)
    {
        photons_per_pixel_bw = get_ini_value<Int>(settings, "ints", "bw-raytracing", Int(32));
        if (Float(int(std::log2(Float(photons_per_pixel_bw)))) != std::log2(Float(photons_per_pixel_bw)))
        {
            std::string error = "number of bw photons per pixel should be a power of 2 ";
            throw std::runtime_error(error);
        }
        Status::print_message("Using "+ std::to_string(photons_per_pixel_bw) + " bw rays per pixel");
    }


    ////// READ THE ATMOSPHERIC DATA //////
    Status::print_message("Reading atmospheric input data from NetCDF.");

    Netcdf_file input_nc(case_name + "_input.nc", Netcdf_mode::Read);
    const int nx = input_nc.get_dimension_size("x");
    const int ny = input_nc.get_dimension_size("y");
    const int n_z_in = input_nc.get_dimension_size("z");
    const int n_lay = input_nc.get_dimension_size("lay");

    const int nz = (n_z_in < n_lay) ? n_z_in+1 : n_z_in;

    const int ncol = nx*ny;

    // Read the x,y,z dimensions if raytracing is enabled
    const Array<Float,1> grid_x(input_nc.get_variable<Float>("x", {nx}), {nx});
    const Array<Float,1> grid_y(input_nc.get_variable<Float>("y", {ny}), {ny});
    const Array<Float,1> grid_z(input_nc.get_variable<Float>("z", {n_z_in}), {n_z_in});
    Array<Float,1> z_lev;

    const Float dx = grid_x({2}) - grid_x({1});
    const Float dy = grid_y({2}) - grid_y({1});
    const Float dz = grid_z({2}) - grid_z({1});
    const Vector<Float> grid_d = {dx, dy, dz};

    const int ngrid_x = input_nc.get_variable<Float>("ngrid_x");
    const int ngrid_y = input_nc.get_variable<Float>("ngrid_y");
    const int ngrid_z = input_nc.get_variable<Float>("ngrid_z");
    const Vector<int> kn_grid = {ngrid_x, ngrid_y, ngrid_z};

    // Read the atmospheric fields.
    const Array<Float,2> tot_tau(input_nc.get_variable<Float>("tot_tau", {n_lay, ny, nx}), {ncol, n_lay});
    const Array<Float,2> tot_ssa(input_nc.get_variable<Float>("tot_ssa", {n_lay, ny, nx}), {ncol, n_lay});
    const Array<Float,2> tot_asy(input_nc.get_variable<Float>("tot_asy", {n_lay, ny, nx}), {ncol, n_lay});

    Array<Float,2> cld_tau({ncol, n_lay});
    Array<Float,2> cld_ssa({ncol, n_lay});
    Array<Float,2> cld_asy({ncol, n_lay});

    cld_tau = std::move(input_nc.get_variable<Float>("cld_tau", {n_lay, ny, nx}));
    cld_ssa = std::move(input_nc.get_variable<Float>("cld_ssa", {n_lay, ny, nx}));
    cld_asy = std::move(input_nc.get_variable<Float>("cld_asy", {n_lay, ny, nx}));

    Array<Float,2> aer_tau({ncol, n_lay});
    Array<Float,2> aer_ssa({ncol, n_lay});
    Array<Float,2> aer_asy({ncol, n_lay});

    aer_tau = std::move(input_nc.get_variable<Float>("aer_tau", {n_lay, ny, nx}));
    aer_ssa = std::move(input_nc.get_variable<Float>("aer_ssa", {n_lay, ny, nx}));
    aer_asy = std::move(input_nc.get_variable<Float>("aer_asy", {n_lay, ny, nx}));

    // read albedo, solar angles, and top-of-domain fluxes
    Array<Float,2> sfc_albedo({ncol, 1});
    sfc_albedo = std::move(input_nc.get_variable<Float>("albedo", {ny, nx}));

    const Float zenith_angle = input_nc.get_variable<Float>("sza");
    const Float azimuth_angle = input_nc.get_variable<Float>("azi");
    const Float tod_dir = input_nc.get_variable<Float>("tod_direct");

    Array<Float,1> mu0({ncol});
    mu0.fill(cos(zenith_angle));

    Camera camera;
    if (switch_bw_raytracing)
    {
        Netcdf_group cam_in = input_nc.get_group("camera-settings");
        camera.fov    = cam_in.get_variable<Float>("fov");
        camera.cam_type = int(cam_in.get_variable<Float>("cam_type"));
        camera.position = {cam_in.get_variable<Float>("px"),
                           cam_in.get_variable<Float>("py"),
                           cam_in.get_variable<Float>("pz")};

        camera.nx  = int(cam_in.get_variable<Float>("nx"));
        camera.ny  = int(cam_in.get_variable<Float>("ny"));
        camera.npix = Int(camera.nx * camera.ny);

        camera.setup_rotation_matrix(cam_in.get_variable<Float>("yaw"),
                                     cam_in.get_variable<Float>("pitch"),
                                     cam_in.get_variable<Float>("roll"));
        camera.setup_normal_camera(camera);

        z_lev.set_dims({n_lay+1});
        z_lev = std::move(input_nc.get_variable<Float>("z_lev", {n_lay+1}));
    }

    // output arrays (setting all dimensions even if we only run FW or BW is note very memory efficiency, so deserves conditional dimension assigment at some point
    Array_gpu<Float,2> flux_tod_dn({nx, ny});
    Array_gpu<Float,2> flux_tod_up({nx, ny});
    Array_gpu<Float,2> flux_sfc_dir({nx, ny});
    Array_gpu<Float,2> flux_sfc_dif({nx, ny});
    Array_gpu<Float,2> flux_sfc_up({nx, ny});
    Array_gpu<Float,3> flux_abs_dir({nx, ny, nz});
    Array_gpu<Float,3> flux_abs_dif({nx, ny, nz});
    Array_gpu<Float,2> radiance({camera.nx, camera.ny});

    Array_gpu<Float,2> flux_dn_2stream;
    Array_gpu<Float,2> flux_up_2stream;
    Array_gpu<Float,2> flux_dn_dir_2stream;

    // empty arrays (mie scattering not (yet) supported in lite version)
    Array<Float,2> mie_cdfs_c;
    Array<Float,3> mie_angs_c;
    Array<Float,3> mie_cdfs_bw_c;
    Array<Float,4> mie_angs_bw_c;
    Array<Float,4> mie_phase_bw_c;
    Array<Float,1> mie_phase_angs_bw_c;
    Array<Float,2> rel_c({ncol, n_lay});

    if (switch_cloud_mie)
    {
       // lwp.set_dims({n_col, n_lay});
       // lwp = std::move(input_nc.get_variable<Float>("lwp", {n_lay, n_col_y, n_col_x}));

        const int n_re  = input_nc.get_dimension_size("r_eff");
        const int n_mie = input_nc.get_dimension_size("n_ang");

        mie_cdfs_c.set_dims({n_mie, 1});
        mie_angs_c.set_dims({n_mie, n_re, 1});
        mie_cdfs_bw_c.set_dims({n_mie, 1, 1});
        mie_angs_bw_c.set_dims({n_mie, n_re, 1, 1});
        mie_phase_bw_c.set_dims({n_mie, n_re, 1, 1});
        mie_phase_angs_bw_c.set_dims({n_mie});

        mie_cdfs_c = std::move(input_nc.get_variable<Float>("phase_cdf", {1, 1, n_mie}));
        mie_angs_c = std::move(input_nc.get_variable<Float>("phase_cdf_angle", {1, 1, n_re, n_mie}));
        mie_cdfs_bw_c = std::move(input_nc.get_variable<Float>("phase_cdf", {1, 1, n_mie}));
        mie_angs_bw_c = std::move(input_nc.get_variable<Float>("phase_cdf_angle", {1, 1, n_re, n_mie}));
        mie_phase_bw_c = std::move(input_nc.get_variable<Float>("phase", {1, 1, n_re, n_mie}));
        mie_phase_angs_bw_c = std::move(input_nc.get_variable<Float>("phase_angle", {n_mie}));

        rel_c = std::move(input_nc.get_variable<Float>("rel", {n_lay, ny, nx}));
    }
    else
    {
        rel_c.fill(Float(0.));
    }

    Array_gpu<Float,2> rel(rel_c);
    Array_gpu<Float,2> mie_cdfs(mie_cdfs_c);
    Array_gpu<Float,3> mie_angs(mie_angs_c);
    Array_gpu<Float,3> mie_cdfs_bw(mie_cdfs_bw_c);
    Array_gpu<Float,4> mie_angs_bw(mie_angs_bw_c);
    Array_gpu<Float,4> mie_phase_bw(mie_phase_bw_c);
    Array_gpu<Float,1> mie_phase_angs_bw(mie_phase_angs_bw_c);

    Array<Float,1> lum_c({ncol});
    lum_c.fill(Float(1.));
    Array_gpu<Float,1> land_use_map(lum_c);

    //// GPU arrays
    Array_gpu<Float,2> tot_tau_g(tot_tau);
    Array_gpu<Float,2> tot_ssa_g(tot_ssa);
    Array_gpu<Float,2> tot_asy_g(tot_asy);
    Array_gpu<Float,2> cld_tau_g(cld_tau);
    Array_gpu<Float,2> cld_ssa_g(cld_ssa);
    Array_gpu<Float,2> cld_asy_g(cld_asy);
    Array_gpu<Float,2> aer_tau_g(aer_tau);
    Array_gpu<Float,2> aer_ssa_g(aer_ssa);
    Array_gpu<Float,2> aer_asy_g(aer_asy);
    Array_gpu<Float,2> sfc_albedo_g(sfc_albedo);
    Array_gpu<Float,1> mu0_g(mu0);

    ////// CREATE THE OUTPUT FILE //////
    // Create the general dimensions and arrays.
    Status::print_message("Preparing NetCDF output file.");

    Netcdf_file output_nc(case_name + "_output.nc", Netcdf_mode::Create);
    if (switch_raytracing || switch_two_stream)
    {
        output_nc.add_dimension("x", nx);
        output_nc.add_dimension("y", ny);
        output_nc.add_dimension("z", nz);
        output_nc.add_dimension("lev", n_lay+1);
    }
    if (switch_bw_raytracing)
    {
        output_nc.add_dimension("nx", camera.nx);
        output_nc.add_dimension("ny", camera.ny);
    }


    if (switch_two_stream)
    {
        flux_up_2stream.set_dims({ncol, n_lay+1});
        flux_dn_2stream.set_dims({ncol, n_lay+1});
        flux_dn_dir_2stream.set_dims({ncol, n_lay+1});

        Rte_sw_rt rte_sw;
        Rte_solver_kernels_cuda_rt::apply_BC(ncol, n_lay, 0, tod_dir * cos(zenith_angle), flux_dn_dir_2stream.ptr());
        Rte_solver_kernels_cuda_rt::apply_BC(ncol, n_lay, 0, flux_dn_2stream.ptr());

        Rte_solver_kernels_cuda_rt::sw_solver_2stream(
            ncol, n_lay, 0,
            tot_tau_g.ptr(),
            tot_ssa_g.ptr(),
            tot_asy_g.ptr(),
            mu0_g.ptr(),
            sfc_albedo_g.ptr(), sfc_albedo_g.ptr(),
            flux_up_2stream.ptr(), flux_dn_2stream.ptr(), flux_dn_dir_2stream.ptr());

        Array<Float,2> flux_up_2stream_c(flux_up_2stream);
        Array<Float,2> flux_dn_2stream_c(flux_dn_2stream);
        Array<Float,2> flux_dn_dir_2stream_c(flux_dn_dir_2stream);

        auto nc_up_2stream = output_nc.add_variable<Float>("flux_up_2stream" , {"lev", "y", "x"});
        auto nc_dn_2stream = output_nc.add_variable<Float>("flux_dn_2stream" , {"lev", "y", "x"});
        auto nc_dn_dir_2stream = output_nc.add_variable<Float>("flux_dn_dir_2stream" , {"lev", "y", "x"});

        nc_up_2stream.insert(flux_up_2stream_c  .v(), {0, 0, 0});
        nc_dn_2stream.insert(flux_dn_2stream_c  .v(), {0, 0, 0});
        nc_dn_dir_2stream.insert(flux_dn_dir_2stream_c  .v(), {0, 0, 0});
    }

    if (switch_raytracing)
    {
        Raytracer raytracer;
        const Vector<int> grid_cells = {nx, ny, nz};

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
                   mie_cdfs,
                   mie_angs,
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
                   tod_dir * std::cos(zenith_angle),
                   Float(0.),
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
    }

    if (switch_bw_raytracing)
    {
        Raytracer_bw raytracer_bw;
        const Vector<int> grid_cells = {nx, ny, n_z_in};

        // Solve the radiation.
        Status::print_message("Starting the backward raytracer!!");

        auto run_solver = [&]()
        {
            cudaDeviceSynchronize();
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);

            raytracer_bw.trace_rays_bb(
                    0,
                    photons_per_pixel_bw, n_lay,
                    grid_cells, grid_d, kn_grid,
                    z_lev,
                    mie_cdfs_bw,
                    mie_angs_bw,
                    mie_phase_bw,
                    mie_phase_angs_bw,
                    rel,
                    tot_tau_g,
                    tot_ssa_g,
                    cld_tau_g,
                    cld_ssa_g,
                    cld_asy_g,
                    aer_tau_g,
                    aer_ssa_g,
                    aer_asy_g,
                    sfc_albedo_g,
                    land_use_map,
                    zenith_angle,
                    azimuth_angle,
                    tod_dir,
                    camera,
                    radiance);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float duration = 0.f;
            cudaEventElapsedTime(&duration, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            Status::print_message("Duration bw raytracer: " + std::to_string(duration) + " (ms)");
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
        Array<Float,2> radiance_c(radiance);
        // Store the output.
        Status::print_message("Storing the bw raytracer output.");

        auto nc_radiance = output_nc.add_variable<Float>("radiance" , {"ny", "nx"});

        nc_radiance.insert(radiance_c  .v(), {0, 0});
    }
    Status::print_message("###### Finished raytracing ######");
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
