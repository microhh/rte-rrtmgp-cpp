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
#include "raytracer_sw.h"
#include "raytracer_lw.h"
#include "raytracer_bw.h"
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

    ////// READ INI FILE //////
    // use mie scattering. Currently only for shortwave and assumes only liquid water is present
    const bool switch_cloud_mie             = get_ini_value<bool>(settings, "clouds", "cloud_mie", false);

    //// Shortwave ////
    const bool switch_shortwave             = get_ini_value<bool>(settings, "shortwave", "shortwave", true);

    // compute and output ray tracer fluxes
    const bool switch_sw_raytracing         = get_ini_value<bool>(settings, "shortwave", "raytracing", true);

    // compute and output plane parallel 1D fluxes (two-stream)
    const bool switch_sw_plane_parallel     = get_ini_value<bool>(settings, "shortwave", "plane_parallel", true);

    // solve ray tracer in independent column mode
    const bool switch_sw_independent_column = get_ini_value<bool>(settings, "shortwave", "rt_independent_column", false);

    //// Longwave ////
    const bool switch_longwave              = get_ini_value<bool>(settings, "longwave", "longwave", true);

    // compute and output ray tracer fluxes
    const bool switch_lw_raytracing         = get_ini_value<bool>(settings, "longwave", "raytracing", true);

    // compute and output plane parallel 1D fluxes (two-stream or no-scattering solution)
    const bool switch_lw_plane_parallel     = get_ini_value<bool>(settings, "longwave", "plane_parallel", true);

    // solve ray tracer in independent column mode
    const bool switch_lw_independent_column = get_ini_value<bool>(settings, "longwave", "rt_independent_column", false);

    // minimum ratio between the lowest gaseous mean free path and the horizontal grid spacing at which ray tracer is still used. Set to 0. to use ray tracer for all g-points
    const Float min_mfp_grid_ratio    = get_ini_value<Float>(settings, "longwave", "min_mfp_grid_ratio", Float(1.0));

    //// Backward (shortwave only) ////
    const bool switch_bw_raytracing         = get_ini_value<bool>(settings, "backward", "bw_raytracing", true);

    // read surface type from land_use_map variable and compute spectral albedo and reflection type (lambertian/specular) accordingly
    const bool switch_lu_albedo             = get_ini_value<bool>(settings, "backward", "lu_albedo", false);

    // solve visible bands and convert to XYZ tristimulus values
    const bool switch_image                 = get_ini_value<bool>(settings, "backward", "image", true);

    // solve broadband radiances
    const bool switch_broadband             = get_ini_value<bool>(settings, "backward", "broadband", false);

    // if >0, overwrite zenith angle from input netcdf file
    Float input_sza = get_ini_value<Float>(settings, "solar_angles", "sza", -1.0);
    // if >0, overwrite azimuth angle from input netcdf file
    Float input_azi = get_ini_value<Float>(settings, "solar_angles", "azi", -1.0);

    Camera camera;
    if (switch_bw_raytracing)
    {
        camera.fov = get_ini_value<Float>(settings, "camera", "cam_field_of_view", 80.);

        // camera type: (0) fish eye camera, (1) rectangular camera, (2) top-of-atmosphere upwelling radiances
        camera.cam_type = get_ini_value<int>(settings, "camera", "cam_type", 0);

        // x,y,z positions of virtual camera
        camera.position = {get_ini_value<Float>(settings, "camera", "cam_px", 0.),
                           get_ini_value<Float>(settings, "camera", "cam_py", 0.),
                           get_ini_value<Float>(settings, "camera", "cam_pz", 100.)};

        // width, height (pixels) of virtual camera or number of zenith and azimuth angles of fish camera
        camera.nx = get_ini_value<int>(settings, "camera", "cam_nx", 0);
        camera.ny = get_ini_value<int>(settings, "camera", "cam_ny", 0);

        // yaw, pitch and roll angles (degrees) of the virtual camera
        camera.setup_rotation_matrix(get_ini_value<Float>(settings, "camera", "cam_yaw", 0.),
                                     get_ini_value<Float>(settings, "camera", "cam_pitch", 0.),
                                     get_ini_value<Float>(settings, "camera", "cam_roll", 0.));
        camera.setup_normal_camera(camera);

        camera.npix = Int(camera.nx * camera.ny);
    }

    // read samples counts (if applicable)
    Int sw_photons_per_pixel;
    if (switch_sw_raytracing)
    {
        sw_photons_per_pixel = get_ini_value<Int>(settings, "shortwave", "samples", Int(256));
        if (Float(int(std::log2(Float(sw_photons_per_pixel)))) != std::log2(Float(sw_photons_per_pixel)))
        {
            std::string error = "number of photons per pixel should be a power of 2 ";
            throw std::runtime_error(error);
        }
    }

    Int bw_photons_per_pixel;
    if (switch_bw_raytracing)
        bw_photons_per_pixel = get_ini_value<Int>(settings, "backward", "samples", Int(1));

    Int lw_photon_power;
    Int lw_photon_count;
    if (switch_lw_raytracing)
    {
        lw_photon_power = get_ini_value<Int>(settings, "longwave", "samples", Int(22));
        lw_photon_count = Int(1) << lw_photon_power;
    }

    if (switch_cloud_mie)
        Status::print_warning("Enabling cloud mie assumes all clouds are liquid (no ice)");


    ////// READ THE ATMOSPHERIC DATA //////
    Status::print_message("Reading atmospheric input data from NetCDF.");

    Netcdf_file input_nc(case_name + "_input.nc", Netcdf_mode::Read);
    const int nx = input_nc.get_dimension_size("x");
    const int ny = input_nc.get_dimension_size("y");
    const int n_z_in = input_nc.get_dimension_size("z");
    const int nlay = input_nc.get_dimension_size("lay");
    const int nlev = input_nc.get_dimension_size("lev");

    const int nz = (n_z_in < nlay) ? n_z_in+1 : n_z_in;

    const int ncol = nx*ny;

    // Read the x,y,z dimensions if raytracing is enabled
    const Array<Float,1> grid_x(input_nc.get_variable<Float>("x", {nx}), {nx});
    const Array<Float,1> grid_y(input_nc.get_variable<Float>("y", {ny}), {ny});
    const Array<Float,1> grid_z(input_nc.get_variable<Float>("z", {n_z_in}), {n_z_in});

    const Array<Float,1> z_lev(input_nc.get_variable<Float>("lev", {nlev}), {nlev});

    const Float dx = grid_x({2}) - grid_x({1});
    const Float dy = grid_y({2}) - grid_y({1});
    const Float dz = grid_z({2}) - grid_z({1});
    const Vector<Float> grid_d = {dx, dy, dz};

    const int ngrid_x = input_nc.get_variable<Float>("ngrid_x");
    const int ngrid_y = input_nc.get_variable<Float>("ngrid_y");
    const int ngrid_z = input_nc.get_variable<Float>("ngrid_z");
    const Vector<int> kn_grid = {ngrid_x, ngrid_y, ngrid_z};

    // Read the atmospheric fields.
    Array<Float,2> sw_tau_tot, sw_ssa_tot, sw_asy_tot;
    Array<Float,2> sw_tau_cld, sw_ssa_cld, sw_asy_cld;
    Array<Float,2> sw_tau_aer, sw_ssa_aer, sw_asy_aer;

    if (switch_shortwave)
    {
        sw_tau_tot.set_dims({ncol, nlay});
        sw_ssa_tot.set_dims({ncol, nlay});
        sw_asy_tot.set_dims({ncol, nlay});
        sw_tau_cld.set_dims({ncol, nlay});
        sw_ssa_cld.set_dims({ncol, nlay});
        sw_asy_cld.set_dims({ncol, nlay});
        sw_tau_aer.set_dims({ncol, nlay});
        sw_ssa_aer.set_dims({ncol, nlay});
        sw_asy_aer.set_dims({ncol, nlay});

        sw_tau_tot = std::move(input_nc.get_variable<Float>("sw_tau_tot", {nlay, ny, nx}));
        sw_ssa_tot = std::move(input_nc.get_variable<Float>("sw_ssa_tot", {nlay, ny, nx}));
        sw_asy_tot = std::move(input_nc.get_variable<Float>("sw_asy_tot", {nlay, ny, nx}));

        sw_tau_cld = std::move(input_nc.get_variable<Float>("sw_tau_cld", {nlay, ny, nx}));
        sw_ssa_cld = std::move(input_nc.get_variable<Float>("sw_ssa_cld", {nlay, ny, nx}));
        sw_asy_cld = std::move(input_nc.get_variable<Float>("sw_asy_cld", {nlay, ny, nx}));

        sw_tau_aer = std::move(input_nc.get_variable<Float>("sw_tau_aer", {nlay, ny, nx}));
        sw_ssa_aer = std::move(input_nc.get_variable<Float>("sw_ssa_aer", {nlay, ny, nx}));
        sw_asy_aer = std::move(input_nc.get_variable<Float>("sw_asy_aer", {nlay, ny, nx}));
    }

    Array<Float,2> lw_tau_tot, lw_ssa_tot, lw_asy_tot;
    Array<Float,2> lw_tau_cld, lw_ssa_cld, lw_asy_cld;
    Array<Float,2> lw_tau_aer, lw_ssa_aer, lw_asy_aer;
    Array<Float,2> source_lay, source_lev;
    Array<Float,1> source_sfc;

    if (switch_longwave)
    {
        lw_tau_tot.set_dims({ncol, nlay});
        lw_ssa_tot.set_dims({ncol, nlay});
        lw_asy_tot.set_dims({ncol, nlay});
        lw_tau_cld.set_dims({ncol, nlay});
        lw_ssa_cld.set_dims({ncol, nlay});
        lw_asy_cld.set_dims({ncol, nlay});
        lw_tau_aer.set_dims({ncol, nlay});
        lw_ssa_aer.set_dims({ncol, nlay});
        lw_asy_aer.set_dims({ncol, nlay});

        lw_tau_tot = std::move(input_nc.get_variable<Float>("lw_tau_tot", {nlay, ny, nx}));
        lw_ssa_tot = std::move(input_nc.get_variable<Float>("lw_ssa_tot", {nlay, ny, nx}));
        lw_asy_tot = std::move(input_nc.get_variable<Float>("lw_asy_tot", {nlay, ny, nx}));

        lw_tau_cld = std::move(input_nc.get_variable<Float>("lw_tau_cld", {nlay, ny, nx}));
        lw_ssa_cld = std::move(input_nc.get_variable<Float>("lw_ssa_cld", {nlay, ny, nx}));
        lw_asy_cld = std::move(input_nc.get_variable<Float>("lw_asy_cld", {nlay, ny, nx}));

        lw_tau_aer = std::move(input_nc.get_variable<Float>("lw_tau_aer", {nlay, ny, nx}));
        lw_ssa_aer = std::move(input_nc.get_variable<Float>("lw_ssa_aer", {nlay, ny, nx}));
        lw_asy_aer = std::move(input_nc.get_variable<Float>("lw_asy_aer", {nlay, ny, nx}));

        source_lay.set_dims({ncol, nlay});
        source_sfc.set_dims({ncol});

        source_lay = std::move(input_nc.get_variable<Float>("source_lay", {nlay, ny, nx}));
        source_sfc = std::move(input_nc.get_variable<Float>("source_sfc", {ny, nx}));

        if (switch_lw_plane_parallel)
        {
            source_lev.set_dims({ncol, nlev});
            source_lev = std::move(input_nc.get_variable<Float>("source_lev", {nlev, ny, nx}));
        }
    }

    // read albedo, solar angles, and top-of-domain fluxes
    // Although solar angles and incoming fluxes should be provided as 2D arrays, ray tracer does not support variables solar angles yet. Currently, only first values of mu0 and azi are used.
    Array<Float,2> alb_sfc;
    Array<Float,2> mu0;
    Array<Float,2> azi;
    Array<Float,2> sw_inc_flux_direct;
    Array<Float,2> sw_inc_flux_diffuse;
    Array<Float,2> lw_inc_flux;
    Array<Float,2> emis_sfc;

    if (switch_shortwave || switch_bw_raytracing)
    {
        alb_sfc.set_dims({nx, ny});
        alb_sfc = input_nc.get_variable<Float>("alb_sfc", {ny, nx});

        mu0.set_dims({nx, ny});
        if (input_sza >= 0)
            mu0.fill(cos(input_sza / Float(180.0) * M_PI));
        else
            mu0 = input_nc.get_variable<Float>("mu0", {ny, nx});

        azi.set_dims({nx, ny});
        if (input_azi >= 0)
            azi.fill(input_azi / Float(180.0) * M_PI);
        else
            azi = input_nc.get_variable<Float>("azi", {ny, nx});

        sw_inc_flux_direct.set_dims({nx, ny});
        sw_inc_flux_direct = input_nc.get_variable<Float>("sw_inc_flux_direct", {ny, nx});

        sw_inc_flux_diffuse.set_dims({nx, ny});
        sw_inc_flux_diffuse = input_nc.get_variable<Float>("sw_inc_flux_diffuse", {ny, nx});

    }

    if (switch_longwave)
    {
        emis_sfc.set_dims({nx, ny});
        emis_sfc = std::move(input_nc.get_variable<Float>("emis_sfc", {ny, nx}));

        lw_inc_flux.set_dims({nx, ny});
        lw_inc_flux = std::move(input_nc.get_variable<Float>("lw_inc_flux", {ny, nx}));
    }

    Array_gpu<Float,2> radiance({camera.nx, camera.ny});

    Array<Float,2> mie_cdfs_c;
    Array<Float,3> mie_angs_c;
    Array<Float,3> mie_cdfs_bw_c;
    Array<Float,4> mie_angs_bw_c;
    Array<Float,4> mie_phase_bw_c;
    Array<Float,1> mie_phase_angs_bw_c;
    Array<Float,2> rel_c({ncol, nlay});

    if (switch_cloud_mie)
    {
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

        rel_c = std::move(input_nc.get_variable<Float>("rel", {nlay, ny, nx}));
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

    ///// GPU arrays
    Array_gpu<Float,2> sw_tau_tot_g(sw_tau_tot);
    Array_gpu<Float,2> sw_ssa_tot_g(sw_ssa_tot);
    Array_gpu<Float,2> sw_asy_tot_g(sw_asy_tot);
    Array_gpu<Float,2> sw_tau_cld_g(sw_tau_cld);
    Array_gpu<Float,2> sw_ssa_cld_g(sw_ssa_cld);
    Array_gpu<Float,2> sw_asy_cld_g(sw_asy_cld);
    Array_gpu<Float,2> sw_tau_aer_g(sw_tau_aer);
    Array_gpu<Float,2> sw_ssa_aer_g(sw_ssa_aer);
    Array_gpu<Float,2> sw_asy_aer_g(sw_asy_aer);

    Array_gpu<Float,2> lw_tau_tot_g(lw_tau_tot);
    Array_gpu<Float,2> lw_ssa_tot_g(lw_ssa_tot);
    Array_gpu<Float,2> lw_asy_tot_g(lw_asy_tot);
    Array_gpu<Float,2> lw_tau_cld_g(lw_tau_cld);
    Array_gpu<Float,2> lw_ssa_cld_g(lw_ssa_cld);
    Array_gpu<Float,2> lw_asy_cld_g(lw_asy_cld);
    Array_gpu<Float,2> lw_tau_aer_g(lw_tau_aer);
    Array_gpu<Float,2> lw_ssa_aer_g(lw_ssa_aer);
    Array_gpu<Float,2> lw_asy_aer_g(lw_asy_aer);
    Array_gpu<Float,2> source_lay_g(source_lay);
    Array_gpu<Float,2> source_lev_g(source_lev);
    Array_gpu<Float,1> source_sfc_g(source_sfc);

    Array_gpu<Float,2> mu0_g(mu0);
    Array_gpu<Float,2> alb_sfc_g(alb_sfc);
    Array_gpu<Float,2> emis_sfc_g(emis_sfc);

    ////// CREATE THE OUTPUT FILE //////
    // Create the general dimensions and arrays.
    Status::print_message("Preparing NetCDF output file.");

    Netcdf_file output_nc(case_name + "_output.nc", Netcdf_mode::Create);

    output_nc.add_dimension("x", nx);
    output_nc.add_dimension("y", ny);
    output_nc.add_dimension("z", n_z_in);
    output_nc.add_dimension("lay", nlay);
    output_nc.add_dimension("lev", nlev);

    Netcdf_group nc_grp_forward;
    Netcdf_group nc_grp_backward;
    Netcdf_group nc_grp_planeparallel;

    if ((switch_shortwave && switch_sw_raytracing) || (switch_longwave && switch_lw_raytracing))
        nc_grp_forward = output_nc.add_group("rt_forward");

    if (switch_bw_raytracing)
        nc_grp_backward = output_nc.add_group("rt_backward");

    if ((switch_shortwave && switch_sw_plane_parallel) || (switch_longwave && switch_lw_plane_parallel))
        nc_grp_planeparallel = output_nc.add_group("plane_parallel");

    auto nc_x = output_nc.add_variable<Float>("x", {"x"});
    auto nc_y = output_nc.add_variable<Float>("y", {"y"});
    auto nc_z = output_nc.add_variable<Float>("z", {"z"});
    nc_x.insert(grid_x.v(), {0});
    nc_y.insert(grid_y.v(), {0});
    nc_z.insert(grid_z.v(), {0});

    if (switch_bw_raytracing)
    {
        output_nc.add_dimension("px", camera.nx);
        output_nc.add_dimension("py", camera.ny);
    }

    ////// longwave solvers
    if (switch_longwave)
    {
        // are scattering properties provided (nonzero)?
        const Bool do_lw_scattering = lw_ssa_tot.max() > Float(0.);

        const Bool bg_profile_present = n_z_in < nlay;

        // output arrays
        Array_gpu<Float,2> lw_rt_flux_tod_dn;
        Array_gpu<Float,2> lw_rt_flux_tod_up;
        Array_gpu<Float,2> lw_rt_flux_sfc_dn;
        Array_gpu<Float,2> lw_rt_flux_sfc_up;
        Array_gpu<Float,3> lw_rt_flux_abs;

        Array_gpu<Float,2> lw_pp_flux_dn;
        Array_gpu<Float,2> lw_pp_flux_up;
        Array_gpu<Float,2> lw_pp_flux_dn_dir;

        if (switch_lw_plane_parallel || bg_profile_present)
        {
            lw_pp_flux_up.set_dims({ncol, nlev});
            lw_pp_flux_dn.set_dims({ncol, nlev});
            lw_pp_flux_dn_dir.set_dims({ncol, nlev});

            Rte_solver_kernels_cuda_rt::apply_BC(ncol, nlay, 0, lw_inc_flux({1,1}), lw_pp_flux_dn.ptr());

            if (do_lw_scattering)
            {
                Rte_solver_kernels_cuda_rt::lw_solver_2stream(
                        ncol, nlay, 0,
                        lw_tau_tot_g.ptr(),
                        lw_ssa_tot_g.ptr(),
                        lw_asy_tot_g.ptr(),
                        source_lay_g.ptr(),
                        source_lev_g.ptr(),
                        emis_sfc_g.ptr(), source_sfc_g.ptr(),
                        lw_pp_flux_up.ptr(), lw_pp_flux_dn.ptr());
            }
            else
            {
                Array_gpu<Float,1> sfc_src_jac(source_sfc.get_dims());
                Array_gpu<Float,2> flux_up_jac(lw_pp_flux_up.get_dims());

                Array_gpu<Float,2> gauss_Ds(Array<Float,2>({1./0.6096748751}, {1,1}));
                Array_gpu<Float,2> gauss_wts(Array<Float,2>({1.}, {1,1}));

                Rte_solver_kernels_cuda_rt::lw_solver_noscat_gaussquad(
                        ncol, nlay, 0, 1,
                        gauss_Ds.ptr(), gauss_wts.ptr(),
                        lw_tau_tot_g.ptr(),
                        source_lay_g.ptr(),
                        source_lev_g.ptr(),
                        emis_sfc_g.ptr(), source_sfc_g.ptr(),
                        lw_pp_flux_up.ptr(), lw_pp_flux_dn.ptr(),
                        sfc_src_jac.ptr(), flux_up_jac.ptr());
            }

            Array<Float,2> lw_pp_flux_up_c(lw_pp_flux_up);
            Array<Float,2> lw_pp_flux_dn_c(lw_pp_flux_dn);

            auto nc_lw_up = nc_grp_planeparallel.add_variable<Float>("lw_flux_up" , {"lev", "y", "x"});
            auto nc_lw_dn = nc_grp_planeparallel.add_variable<Float>("lw_flux_dn" , {"lev", "y", "x"});

            nc_lw_up.insert(lw_pp_flux_up_c  .v(), {0, 0, 0});
            nc_lw_dn.insert(lw_pp_flux_dn_c  .v(), {0, 0, 0});
        }

        if (switch_lw_raytracing)
        {
            lw_rt_flux_tod_dn.set_dims({nx, ny});
            lw_rt_flux_tod_up.set_dims({nx, ny});
            lw_rt_flux_sfc_dn.set_dims({nx, ny});
            lw_rt_flux_sfc_up.set_dims({nx, ny});
            lw_rt_flux_abs.set_dims({nx, ny, n_z_in});

            Raytracer_lw raytracer_lw;
            const Vector<int> grid_cells = {nx, ny, n_z_in};

            const Float lw_flux_tod = bg_profile_present ? lw_pp_flux_dn({1, grid_cells.z+1}) : Float(0.);

            Status::print_message("Starting the longwave raytracer!!");

            cudaDeviceSynchronize();
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);
            // do something.

	        raytracer_lw.trace_rays(
                   0,
                   switch_lw_independent_column,
                   lw_photon_count,
                   grid_cells,
                   grid_d,
                   kn_grid,
                   lw_tau_tot_g,
                   lw_ssa_tot_g,
                   lw_tau_cld_g,
                   lw_ssa_cld_g,
                   lw_asy_cld_g,
                   lw_tau_aer_g,
                   lw_ssa_aer_g,
                   lw_asy_aer_g,
                   source_lay_g,
                   source_sfc_g,
                   emis_sfc_g,
                   lw_flux_tod,
                   lw_rt_flux_tod_dn,
                   lw_rt_flux_tod_up,
                   lw_rt_flux_sfc_dn,
                   lw_rt_flux_sfc_up,
                   lw_rt_flux_abs);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float duration = 0.f;
            cudaEventElapsedTime(&duration, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            Status::print_message("Duration longwave raytracer: " + std::to_string(duration) + " (ms)");

            // output arrays to cpu
            Array<Float,2> flux_tod_dn_c(lw_rt_flux_tod_dn);
            Array<Float,2> flux_tod_up_c(lw_rt_flux_tod_up);
            Array<Float,2> flux_sfc_dn_c(lw_rt_flux_sfc_dn);
            Array<Float,2> flux_sfc_up_c(lw_rt_flux_sfc_up);
            Array<Float,3> flux_abs_c(lw_rt_flux_abs);

            // Store the output.
            Status::print_message("Storing the longwave raytracer output.");

            auto nc_flux_tod_dn     = nc_grp_forward.add_variable<Float>("lw_flux_tod_dn" , {"y", "x"});
            auto nc_flux_tod_up     = nc_grp_forward.add_variable<Float>("lw_flux_tod_up" , {"y", "x"});
            auto nc_flux_sfc_dn     = nc_grp_forward.add_variable<Float>("lw_flux_sfc_dn" , {"y", "x"});
            auto nc_flux_sfc_up     = nc_grp_forward.add_variable<Float>("lw_flux_sfc_up" , {"y", "x"});
            auto nc_flux_abs        = nc_grp_forward.add_variable<Float>("lw_abs"         , {"z", "y", "x"});

            nc_flux_tod_dn   .insert(flux_tod_dn_c  .v(), {0, 0});
            nc_flux_tod_up   .insert(flux_tod_up_c  .v(), {0, 0});
            nc_flux_sfc_dn   .insert(flux_sfc_dn_c  .v(), {0, 0});
            nc_flux_sfc_up   .insert(flux_sfc_up_c  .v(), {0, 0});
            nc_flux_abs      .insert(flux_abs_c     .v(), {0, 0, 0});
        }
    }

    ////// shortwave solvers
    if (switch_shortwave)
    {
        // output arrays
        Array_gpu<Float,2> sw_rt_flux_tod_dn;
        Array_gpu<Float,2> sw_rt_flux_tod_up;
        Array_gpu<Float,2> sw_rt_flux_sfc_dir;
        Array_gpu<Float,2> sw_rt_flux_sfc_dif;
        Array_gpu<Float,2> sw_rt_flux_sfc_up;
        Array_gpu<Float,3> sw_rt_flux_abs_dir;
        Array_gpu<Float,3> sw_rt_flux_abs_dif;

        Array_gpu<Float,2> sw_pp_flux_dn;
        Array_gpu<Float,2> sw_pp_flux_up;
        Array_gpu<Float,2> sw_pp_flux_dn_dir;

        if (switch_sw_plane_parallel)
        {
            sw_pp_flux_up.set_dims({ncol, nlev});
            sw_pp_flux_dn.set_dims({ncol, nlev});
            sw_pp_flux_dn_dir.set_dims({ncol, nlev});

            Rte_solver_kernels_cuda_rt::apply_BC(ncol, nlay, 0, sw_inc_flux_direct({1,1}), sw_pp_flux_dn_dir.ptr());
            Rte_solver_kernels_cuda_rt::apply_BC(ncol, nlay, 0, sw_inc_flux_diffuse({1,1}), sw_pp_flux_dn.ptr());

            Rte_solver_kernels_cuda_rt::sw_solver_2stream(
                ncol, nlay, 0,
                sw_tau_tot_g.ptr(),
                sw_ssa_tot_g.ptr(),
                sw_asy_tot_g.ptr(),
                mu0_g.ptr(),
                alb_sfc_g.ptr(), alb_sfc_g.ptr(),
                sw_pp_flux_up.ptr(), sw_pp_flux_dn.ptr(), sw_pp_flux_dn_dir.ptr());

            Array<Float,2> sw_pp_flux_up_c(sw_pp_flux_up);
            Array<Float,2> sw_pp_flux_dn_c(sw_pp_flux_dn);
            Array<Float,2> sw_pp_flux_dn_dir_c(sw_pp_flux_dn_dir);

            auto nc_sw_up = nc_grp_planeparallel.add_variable<Float>("sw_flux_up" , {"lev", "y", "x"});
            auto nc_sw_dn = nc_grp_planeparallel.add_variable<Float>("sw_flux_dn" , {"lev", "y", "x"});
            auto nc_sw_dn_dir = nc_grp_planeparallel.add_variable<Float>("sw_flux_dn_dir" , {"lev", "y", "x"});

            nc_sw_up.insert(sw_pp_flux_up_c  .v(), {0, 0, 0});
            nc_sw_dn.insert(sw_pp_flux_dn_c  .v(), {0, 0, 0});
            nc_sw_dn_dir.insert(sw_pp_flux_dn_dir_c  .v(), {0, 0, 0});
        }

        if (switch_sw_raytracing)
        {
            sw_rt_flux_tod_dn.set_dims({nx, ny});
            sw_rt_flux_tod_up.set_dims({nx, ny});
            sw_rt_flux_sfc_dir.set_dims({nx, ny});
            sw_rt_flux_sfc_dif.set_dims({nx, ny});
            sw_rt_flux_sfc_up.set_dims({nx, ny});
            sw_rt_flux_abs_dir.set_dims({nx, ny, nz});
            sw_rt_flux_abs_dif.set_dims({nx, ny, nz});

            Raytracer raytracer;
            const Vector<int> grid_cells = {nx, ny, nz};

            // Solve the radiation.
            Status::print_message("Starting the shortwave raytracer!!");

            cudaDeviceSynchronize();
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);
            // do something.

            raytracer.trace_rays(
                   0,
                   switch_sw_independent_column,
                   sw_photons_per_pixel,
                   grid_cells,
                   grid_d,
                   kn_grid,
                   mie_cdfs,
                   mie_angs,
                   sw_tau_tot_g,
                   sw_ssa_tot_g,
                   sw_tau_cld_g,
                   sw_ssa_cld_g,
                   sw_asy_cld_g,
                   sw_tau_aer_g,
                   sw_ssa_aer_g,
                   sw_asy_aer_g,
                   rel,
                   alb_sfc_g,
                   acos(mu0({1,1})),
                   azi({1,1}),
                   sw_inc_flux_direct({1,1}),
                   sw_inc_flux_diffuse({1,1}),
                   sw_rt_flux_tod_dn,
                   sw_rt_flux_tod_up,
                   sw_rt_flux_sfc_dir,
                   sw_rt_flux_sfc_dif,
                   sw_rt_flux_sfc_up,
                   sw_rt_flux_abs_dir,
                   sw_rt_flux_abs_dif);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float duration = 0.f;
            cudaEventElapsedTime(&duration, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            Status::print_message("Duration shortwave raytracer: " + std::to_string(duration) + " (ms)");

            // output arrays to cpu
            Array<Float,2> flux_tod_dn_c(sw_rt_flux_tod_dn);
            Array<Float,2> flux_tod_up_c(sw_rt_flux_tod_up);
            Array<Float,2> flux_sfc_dir_c(sw_rt_flux_sfc_dir);
            Array<Float,2> flux_sfc_dif_c(sw_rt_flux_sfc_dif);
            Array<Float,2> flux_sfc_up_c(sw_rt_flux_sfc_up);
            Array<Float,3> flux_abs_dir_c(sw_rt_flux_abs_dir);
            Array<Float,3> flux_abs_dif_c(sw_rt_flux_abs_dif);
            // Store the output.
            Status::print_message("Storing the shortwave raytracer output.");

            auto nc_flux_tod_dn     = nc_grp_forward.add_variable<Float>("sw_flux_tod_dn" , {"y", "x"});
            auto nc_flux_tod_up     = nc_grp_forward.add_variable<Float>("sw_flux_tod_up" , {"y", "x"});
            auto nc_flux_sfc_dir    = nc_grp_forward.add_variable<Float>("sw_flux_sfc_dir", {"y", "x"});
            auto nc_flux_sfc_dif    = nc_grp_forward.add_variable<Float>("sw_flux_sfc_dif", {"y", "x"});
            auto nc_flux_sfc_up     = nc_grp_forward.add_variable<Float>("sw_flux_sfc_up" , {"y", "x"});
            auto nc_flux_abs_dir    = nc_grp_forward.add_variable<Float>("sw_abs_dir"     , {"z", "y", "x"});
            auto nc_flux_abs_dif    = nc_grp_forward.add_variable<Float>("sw_abs_dif"     , {"z", "y", "x"});

            nc_flux_tod_dn   .insert(flux_tod_dn_c  .v(), {0, 0});
            nc_flux_tod_up   .insert(flux_tod_up_c  .v(), {0, 0});
            nc_flux_sfc_dir  .insert(flux_sfc_dir_c .v(), {0, 0});
            nc_flux_sfc_dif  .insert(flux_sfc_dif_c .v(), {0, 0});
            nc_flux_sfc_up   .insert(flux_sfc_up_c  .v(), {0, 0});
            nc_flux_abs_dir  .insert(flux_abs_dir_c .v(), {0, 0, 0});
            nc_flux_abs_dif  .insert(flux_abs_dif_c .v(), {0, 0, 0});
        }
    }

    ////// backward solver
    if (switch_bw_raytracing)
    {
        Raytracer_bw raytracer_bw;
        const Vector<int> grid_cells = {nx, ny, n_z_in};

        Array_gpu<Float,1> z_lev_g(z_lev);

        // Solve the radiation.
        Status::print_message("Starting the backward raytracer!!");

        cudaDeviceSynchronize();
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        raytracer_bw.trace_rays_bb(
                0,
                bw_photons_per_pixel, nlay,
                grid_cells, grid_d, kn_grid,
                z_lev_g,
                mie_cdfs_bw,
                mie_angs_bw,
                mie_phase_bw,
                mie_phase_angs_bw,
                rel,
                sw_tau_tot_g,
                sw_ssa_tot_g,
                sw_tau_cld_g,
                sw_ssa_cld_g,
                sw_asy_cld_g,
                sw_tau_aer_g,
                sw_ssa_aer_g,
                sw_asy_aer_g,
                alb_sfc_g,
                land_use_map,
                acos(mu0({1,1})),
                azi({1,1}),
                sw_inc_flux_direct({1,1})/mu0({1,1}),
                sw_inc_flux_diffuse({1,1}),
                camera,
                radiance);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float duration = 0.f;
        cudaEventElapsedTime(&duration, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        Status::print_message("Duration bw raytracer: " + std::to_string(duration) + " (ms)");

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
