#include <curand_kernel.h>
#include <iomanip>
#include <iostream>

#include "raytracer_kernels_bw_lw.h"
#include "raytracer_definitions.h"


namespace
{
    using namespace Raytracer_functions;

    constexpr Float w_thres = 0.5;

    template<Bool dz_constant> __device__
    inline void reset_photon(
            Photon& photon,
            Float* __restrict__ camera_shot,
            const Int ij_cam, const int n,
            Random_number_generator<Float>& rng,
            const Grid_knull* __restrict__ k_null_grid,
            const Float* __restrict__ k_ext,
            const Vector<int>& kn_grid,
            const Vector<Float>& kn_grid_d,
            const Vector<Float>& grid_d,
            const Vector<Float>& grid_size,
            const Vector<int>& grid_cells,
            Float& weight, int& bg_idx,
            const Camera& camera,
            const int kbg,
            const Float* __restrict__ z_lev_bg,
            const Float* __restrict__ z_lev,
            const Float* __restrict__ kn_z_lev,
            const int* __restrict__ z_lut,
            const int* __restrict__ kn_z_lut,
            const Float lut_dz,
            const int lut_size,
            const Float s_min,
            const Float s_min_bg)
    {
        const Float i = (Float(ij_cam % camera.nx) + rng())/ Float(camera.nx);
        const Float j = (Float(ij_cam / camera.nx) + rng())/ Float(camera.ny);


        if (camera.cam_type == 0)
        {
            // Fish eye camera
            const Float photon_zenith = i * Float(.5) * camera.fov / Float(180.) * M_PI;
            const Float photon_azimuth = j * Float(2.) * M_PI;
            const Vector<Float> dir_tmp = {cos(photon_zenith), sin(photon_zenith) * cos(photon_azimuth), sin(photon_zenith) * sin(photon_azimuth)};

            photon.direction.x = dot(camera.mx,  dir_tmp);
            photon.direction.y = dot(camera.my,  dir_tmp);
            photon.direction.z = dot(camera.mz,  dir_tmp);
            photon.position = camera.position + s_min;
        }
        else if (camera.cam_type == 1)
        {
            // Rectangular camera based on Villefranque et al. 2019
            photon.direction = normalize(camera.cam_width * (Float(2.)*i-Float(1.0)) + camera.cam_height * (Float(2.)*j-Float(1.0)) + camera.cam_depth);
            photon.position = camera.position + s_min;
        }
        else
        {
            // Top-of-atmosphere upwelling radiances
            photon.direction = {camera.mx.x, camera.my.x, camera.mz.x};
            photon.position.x = (Float(ij_cam % camera.nx) + rng()) * (grid_size.x / camera.nx);
            photon.position.y = (Float(ij_cam / camera.nx) + rng()) * (grid_size.y / camera.ny);
            photon.position.z = z_lev_bg[kbg] - s_min;
        }

        // if camera starts above domain top, bring photon towards domain top
        if ((photon.position.z > z_lev_bg[kbg]) and (photon.direction.z < Float(0.)))
        {
            Float ds = (photon.position.z - z_lev_bg[kbg]) / photon.direction.z;

            photon.position.z = z_lev_bg[kbg] - s_min;
            photon.position.y += photon.direction.y * ds;
            photon.position.x += photon.direction.x * ds;

            // Cyclic boundary condition in x.
            photon.position.x = fmod(photon.position.x, grid_size.x);
            if (photon.position.x < Float(0.))
                photon.position.x += grid_size.x;

            // Cyclic boundary condition in y.
            photon.position.y = fmod(photon.position.y, grid_size.y);
            if (photon.position.y < Float(0.))
                photon.position.y += grid_size.y;
        }

        photon.kind = Photon_kind::Direct;
        weight = 1;
        bg_idx = 0;

        for (int i=0; i<kbg; ++i)
        {
            if (photon.position.z > z_lev_bg[i]) bg_idx = i;
        }

        atomicAdd(&camera_shot[ij_cam], Float(1.));

    }

}

template<Bool dz_constant>__global__
void ray_tracer_kernel_bw_lw(
        const int igpt,
        const Int photons_per_pixel,
        const Grid_knull* __restrict__ k_null_grid,
        Float* __restrict__ camera_count,
        Float* __restrict__ camera_shot,
        Int* __restrict__ counter,
        const Float* __restrict__ k_ext, const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ k_ext_bg, const Optics_scat* __restrict__ scat_asy_bg,
        const Float* __restrict__ z_lev_bg,
        const Float* __restrict__ sfc_source,
        const Float* __restrict__ lay_source,
        const Float* __restrict__ surface_emis,
        const Float tod_incoming_flux,
        const Vector<Float> grid_size, const Vector<Float> grid_d,
        const Vector<int> grid_cells, const Vector<int> kn_grid,
        const Float* __restrict__ z_lev,
        const Float* __restrict__ kn_z_lev,
        const int* __restrict__ z_lut,
        const int* __restrict__ kn_z_lut,
        const Float lut_dz,
        const int lut_size,
        const Camera camera,
        const int kbg)
{
    Vector<Float> surface_normal = {0, 0, 1};

    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    const Vector<Float> kn_grid_d = grid_size / kn_grid;
    const Float z_top = z_lev_bg[kbg];

    Random_number_generator<Float> rng(n+bw_lw_kernel_block*bw_lw_kernel_grid*igpt);

    const Float s_min = max(max(grid_size.z, grid_size.x), grid_size.y) * Float_epsilon;
    const Float s_min_bg = max(max(grid_size.x, grid_size.y), z_top) * Float_epsilon;

    while (counter[0] < camera.npix*photons_per_pixel)
    {
        const Int count = atomicAdd(&counter[0], 1);
        const Int ij_cam = count / photons_per_pixel;

        if (ij_cam >= camera.npix)
            return;

        Float weight;
        int bg_idx;

        Photon photon;

        reset_photon<dz_constant>(
                photon, camera_shot,
                ij_cam, n, rng,
                k_null_grid, k_ext,
                kn_grid, kn_grid_d, grid_d,
                grid_size, grid_cells,
                weight, bg_idx,
                camera,
                kbg, z_lev_bg,
                z_lev, kn_z_lev, z_lut, kn_z_lut, lut_dz, lut_size,
                s_min, s_min_bg);

        Float tau;
        Float d_max = Float(0.);
        Float k_ext_null;
        bool transition = false;
        int i_n, j_n, k_n, ijk_n;

        bool photon_alive = true;
        while (photon_alive)
        {
            if (!transition)
            {
                tau = sample_tau(rng());
            }
            transition = false;

            // 1D raytracing between TOD and TOA?
            if (photon.position.z > grid_size.z)
            {
                const Float dn = max(Float_epsilon, tau / k_ext_bg[bg_idx]);
                d_max = abs( (photon.direction.z>0) ? (z_lev_bg[bg_idx+1] - photon.position.z) / photon.direction.z : (z_lev_bg[bg_idx] - photon.position.z) / photon.direction.z );
                if (dn >= d_max)
                {
                    photon.position.z = (photon.direction.z > 0) ? z_lev_bg[bg_idx+1] + s_min_bg : z_lev_bg[bg_idx] - s_min_bg;
                    photon.position.y += photon.direction.y * d_max;
                    photon.position.x += photon.direction.x * d_max;

                    // move to actual grid: reduce tau and set next position
                    if (photon.position.z <= grid_size.z + s_min_bg)
                    {
                        tau -= k_ext_bg[bg_idx] * (d_max + s_min_bg);
                        photon.position.z = grid_size.z - s_min;
                        d_max = Float(0.);
                        transition=true;

                        // Cyclic boundary condition in x.
                        photon.position.x = fmod(photon.position.x, grid_size.x);
                        if (photon.position.x < Float(0.))
                            photon.position.x += grid_size.x;

                        // Cyclic boundary condition in y.
                        photon.position.y = fmod(photon.position.y, grid_size.y);
                        if (photon.position.y < Float(0.))
                            photon.position.y += grid_size.y;
                    }
                    else if (photon.position.z >= z_top)
                    {
                        atomicAdd(&camera_count[ij_cam], weight * tod_incoming_flux / Float(M_PI));

                        // Leaving top-of-domain
                        photon_alive = false;
                    }
                    else
                    {
                        // just move to next grid
                        transition = true;
                        tau -= k_ext_bg[bg_idx] * (d_max + s_min_bg);

                        bg_idx += (photon.direction.z > 0) ? 1 : -1;
                    }
                }
                else
                {
                    const Float dz = photon.direction.z * dn;
                    photon.position.z = (dz > 0) ? min(photon.position.z + dz, z_lev_bg[bg_idx+1] - s_min_bg) : max(photon.position.z + dz, z_lev_bg[bg_idx] + s_min_bg);

                    photon.position.y += photon.direction.y * dn;
                    photon.position.x += photon.direction.x * dn;

                    // Compute probability not being absorbed and store weighted absorption probability
                    const Float k_sca_bg_tot = scat_asy_bg[bg_idx].k_sca_gas + scat_asy_bg[bg_idx].k_sca_cld + scat_asy_bg[bg_idx].k_sca_aer;
                    const Float ssa_bg_tot = k_sca_bg_tot / k_ext_bg[bg_idx];

                    // TODO BT source function
                    atomicAdd(&camera_count[ij_cam], weight * (Float(1.) - ssa_bg_tot) * lay_source[(grid_cells.z + bg_idx) * grid_cells.x * grid_cells.y]);

                    // Update weights (see Iwabuchi 2006: https://doi.org/10.1175/JAS3755.1)
                    weight *= ssa_bg_tot;
                    if (weight < w_thres)
                        weight = (rng() > weight) ? Float(0.) : Float(1.);

                    // only with nonzero weight continue ray tracing, else start new ray
                    if (weight > Float(0.))
                    {
                        // find scatter type: 0 = gas, 1 = cloud, 2 = aerosol (although we assume clear sky bg profile, so no option for Mie scattering
                        const Float scatter_rng = rng();
                        const int scatter_type = scatter_rng < (scat_asy_bg[bg_idx].k_sca_aer/k_sca_bg_tot) ? 2 :
                                                 scatter_rng < ((scat_asy_bg[bg_idx].k_sca_aer+scat_asy_bg[bg_idx].k_sca_cld)/k_sca_bg_tot) ? 1 : 0;
                        Float g;
                        switch (scatter_type)
                        {
                            case 0:
                                g = Float(0.);
                                break;
                            case 1:
                                g = min(Float(1.) - Float_epsilon, scat_asy_bg[bg_idx].asy_cld);
                                break;
                            case 2:
                                g = min(Float(1.) - Float_epsilon, scat_asy_bg[bg_idx].asy_aer);
                                break;
                        }
                        const Float cos_scat = (scatter_type == 0) ? rayleigh(rng()) : henyey(g, rng());
                        const Float sin_scat = max(Float(0.), sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon));

                        Vector<Float> t1{Float(0.), Float(0.), Float(0.)};
                        if (fabs(photon.direction.x) < fabs(photon.direction.y))
                        {
                            if (fabs(photon.direction.x) < fabs(photon.direction.z))
                                t1.x = Float(1.);
                            else
                                t1.z = Float(1.);
                        }
                        else
                        {
                            if (fabs(photon.direction.y) < fabs(photon.direction.z))
                                t1.y = Float(1.);
                            else
                                t1.z = Float(1.);
                        }
                        t1 = normalize(t1 - photon.direction*dot(t1, photon.direction));
                        Vector<Float> t2 = cross(photon.direction, t1);

                        const Float phi = Float(2.*M_PI)*rng();

                        photon.direction = cos_scat*photon.direction
                                + sin_scat*(sin(phi)*t1 + cos(phi)*t2);

                        photon.kind = Photon_kind::Diffuse;
                    }
                    else
                    {
                        photon_alive = false;
                    }
                }
            }
            // we reached the 'dynamical' domain, now things get interesting
            else
            {
                // if d_max is zero, find current grid and maximum distance
                if (d_max == Float(0.))
                {
                    i_n = float_to_int(photon.position.x, kn_grid_d.x, kn_grid.x);
                    j_n = float_to_int(photon.position.y, kn_grid_d.y, kn_grid.y);
                    k_n = grid_z_index<dz_constant>(photon.position.z, kn_grid_d.z, kn_z_lev, kn_z_lut, lut_dz, lut_size, kn_grid.z);
                    const Float sx = abs((photon.direction.x > 0) ? ((i_n+1) * kn_grid_d.x - photon.position.x)/photon.direction.x : (i_n*kn_grid_d.x - photon.position.x)/photon.direction.x);
                    const Float sy = abs((photon.direction.y > 0) ? ((j_n+1) * kn_grid_d.y - photon.position.y)/photon.direction.y : (j_n*kn_grid_d.y - photon.position.y)/photon.direction.y);
                    const Float sz = abs((photon.direction.z > 0) ? (grid_z_bound<dz_constant>(k_n+1, kn_grid_d.z, kn_z_lev) - photon.position.z)/photon.direction.z : (grid_z_bound<dz_constant>(k_n, kn_grid_d.z, kn_z_lev) - photon.position.z)/photon.direction.z);
                    d_max = min(sx, min(sy, sz));
                    ijk_n = i_n + j_n*kn_grid.x + k_n*kn_grid.y*kn_grid.x;
                    k_ext_null = k_null_grid[ijk_n].k_max;
                }

                const Float dn = max(Float_epsilon, tau / k_ext_null);

                if ( ( dn >= d_max) )
                {
                    const Float dx = photon.direction.x * (d_max);
                    const Float dy = photon.direction.y * (d_max);
                    const Float dz = photon.direction.z * (d_max);

                    photon.position.x += dx;
                    photon.position.y += dy;
                    photon.position.z += dz;

                    // surface hit
                    if (photon.position.z < Float_epsilon)
                    {
                        photon.position.z = Float_epsilon;
                        const int i = float_to_int(photon.position.x, grid_d.x, grid_cells.x);
                        const int j = float_to_int(photon.position.y, grid_d.y, grid_cells.y);
                        const int ij = i + j*grid_cells.x;
                        d_max = Float(0.);

                        atomicAdd(&camera_count[ij_cam], weight * surface_emis[ij] * sfc_source[ij]);

                        weight *= (Float(1.) - surface_emis[ij]);

                        if (weight < w_thres)
                            weight = (rng() > weight) ? Float(0.) : Float(1.);

                        // only with nonzero weight continue ray tracing, else start new ray
                        if (weight > Float(0.))
                        {
                            const Float mu_surface = sqrt(rng());
                            const Float azimuth_surface = Float(2.*M_PI)*rng();
                            const Float sin_theta = sqrt(Float(1.) - mu_surface*mu_surface + Float_epsilon);

                            photon.direction.x = sin_theta*sin(azimuth_surface);
                            photon.direction.y = sin_theta*cos(azimuth_surface);
                            photon.direction.z = mu_surface;
                            photon.kind = Photon_kind::Diffuse;

                        }
                        else
                        {
                            photon_alive = false;
                        }
                    }

                    // TOD exit
                    else if (photon.position.z >= grid_size.z)
                    {
                        // TODO: how to handle TOD exits?
                        photon.position.z = grid_size.z + s_min_bg;
                        tau -= d_max * k_ext_null;
                        bg_idx = 0;
                        d_max = Float(0.);
                        transition = true;

                    }
                    // regular cell crossing: adjust tau and apply periodic BC
                    else
                    {
                        photon.position.x += photon.direction.x>0 ? s_min : -s_min;
                        photon.position.y += photon.direction.y>0 ? s_min : -s_min;
                        photon.position.z += photon.direction.z>0 ? s_min : -s_min;

                        // Cyclic boundary condition in x.
                        photon.position.x = fmod(photon.position.x, grid_size.x);
                        if (photon.position.x < Float(0.))
                            photon.position.x += grid_size.x;

                        // Cyclic boundary condition in y.
                        photon.position.y = fmod(photon.position.y, grid_size.y);
                        if (photon.position.y < Float(0.))
                            photon.position.y += grid_size.y;

                        tau -= d_max * k_ext_null;
                        d_max = Float(0.);
                        transition = true;
                    }
                }
                else
                {
                    const Float dx = photon.direction.x * dn;
                    const Float dy = photon.direction.y * dn;
                    const Float dz = photon.direction.z * dn;

                    photon.position.x = (dx > 0) ? min(photon.position.x + dx, (i_n+1) * kn_grid_d.x - s_min) : max(photon.position.x + dx, (i_n) * kn_grid_d.x + s_min);
                    photon.position.y = (dy > 0) ? min(photon.position.y + dy, (j_n+1) * kn_grid_d.y - s_min) : max(photon.position.y + dy, (j_n) * kn_grid_d.y + s_min);
                    photon.position.z = (dz > 0) ? min(photon.position.z + dz, grid_z_bound<dz_constant>(k_n+1, kn_grid_d.z, kn_z_lev) - s_min) : max(photon.position.z + dz, grid_z_bound<dz_constant>(k_n, kn_grid_d.z, kn_z_lev) + s_min);

                    // Calculate the 3D index.
                    const int i = float_to_int(photon.position.x, grid_d.x, grid_cells.x);
                    const int j = float_to_int(photon.position.y, grid_d.y, grid_cells.y);
                    const int k = grid_z_index<dz_constant>(photon.position.z, grid_d.z, z_lev, z_lut, lut_dz, lut_size, grid_cells.z);
                    const int ijk = i + j*grid_cells.x + k*grid_cells.x*grid_cells.y;


                    // Compute probability not being absorbed and store weighted absorption probability
                    const Float k_sca_tot = scat_asy[ijk].k_sca_gas + scat_asy[ijk].k_sca_cld + scat_asy[ijk].k_sca_aer;
                    const Float ssa_tot = k_sca_tot / k_ext[ijk];
                    const Float f_abs = (Float(1.) - ssa_tot) * (k_ext[ijk]/k_ext_null);

                    atomicAdd(&camera_count[ij_cam], weight * f_abs * lay_source[ijk]);

                    // Update weights (see Iwabuchi 2006: https://doi.org/10.1175/JAS3755.1)
                    weight *= (Float(1.) - f_abs);

                    if (weight < w_thres)
                        weight = (rng() > weight) ? Float(0.) : Float(1.);

                    // only with nonzero weight continue ray tracing, else start new ray
                    if (weight > Float(0.))
                    {
                        // Null collision.
                        if (rng() >= ssa_tot / (ssa_tot - Float(1.) + k_ext_null / k_ext[ijk]))
                        {
                            d_max -= dn;
                        }
                        // Scattering.
                        else
                        {
                            d_max = Float(0.);
                            // find scatter type: 0 = gas, 1 = cloud, 2 = aerosol
                            const Float scatter_rng = rng();
                            const int scatter_type = scatter_rng < (scat_asy[ijk].k_sca_aer/k_sca_tot) ? 2 :
                                                     scatter_rng < ((scat_asy[ijk].k_sca_aer+scat_asy[ijk].k_sca_cld)/k_sca_tot) ? 1 : 0;
                            Float g;
                            switch (scatter_type)
                            {
                                case 0:
                                    g = Float(0.);
                                    break;
                                case 1:
                                    g = min(Float(1.) - Float_epsilon, scat_asy[ijk].asy_cld);
                                    break;
                                case 2:
                                    g = min(Float(1.) - Float_epsilon, scat_asy[ijk].asy_aer);
                                    break;
                            }
                            const Float cos_scat = (scatter_type == 0) ? rayleigh(rng()) : henyey(g, rng());
                            const Float sin_scat = max(Float(0.), sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon));

                            Vector<Float> t1{Float(0.), Float(0.), Float(0.)};
                            if (fabs(photon.direction.x) < fabs(photon.direction.y))
                            {
                                if (fabs(photon.direction.x) < fabs(photon.direction.z))
                                    t1.x = Float(1.);
                                else
                                    t1.z = Float(1.);
                            }
                            else
                            {
                                if (fabs(photon.direction.y) < fabs(photon.direction.z))
                                    t1.y = Float(1.);
                                else
                                    t1.z = Float(1.);
                            }
                            t1 = normalize(t1 - photon.direction*dot(t1, photon.direction));
                            Vector<Float> t2 = cross(photon.direction, t1);

                            const Float phi = Float(2.*M_PI)*rng();

                            photon.direction = cos_scat*photon.direction
                                + sin_scat*(sin(phi)*t1 + cos(phi)*t2);
                            photon.kind = Photon_kind::Diffuse;

                        }
                    }
                    else
                    {
                        photon_alive = false;
                    }
                }
            }
        }
    }
}


void launch_ray_tracer_kernel_bw_lw(
        const bool dz_constant,
        const dim3 grid,
        const dim3 block,
        const int igpt,
        const Int photons_per_pixel,
        const Grid_knull* __restrict__ k_null_grid,
        Float* __restrict__ camera_count,
        Float* __restrict__ camera_shot,
        Int* __restrict__ counter,
        const Float* __restrict__ k_ext, const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ k_ext_bg, const Optics_scat* __restrict__ scat_asy_bg,
        const Float* __restrict__ z_lev_bg,
        const Float* __restrict__ sfc_source,
        const Float* __restrict__ lay_source,
        const Float* __restrict__ surface_emis,
        const Float tod_incoming_flux,
        const Vector<Float> grid_size, const Vector<Float> grid_d,
        const Vector<int> grid_cells, const Vector<int> kn_grid,
        const Float* __restrict__ z_lev,
        const Float* __restrict__ kn_z_lev,
        const int* __restrict__ z_lut,
        const int* __restrict__ kn_z_lut,
        const Float lut_dz,
        const int lut_size,
        const Camera camera,
        const int nbg)
{
    auto launch = [&](auto kernel)
    {
        kernel<<<grid, block>>>(
                igpt,
                photons_per_pixel, k_null_grid,
                camera_count,
                camera_shot,
                counter,
                k_ext, scat_asy,
                k_ext_bg, scat_asy_bg,
                z_lev_bg,
                sfc_source, lay_source, surface_emis,
                tod_incoming_flux,
                grid_size, grid_d, grid_cells, kn_grid,
                z_lev, kn_z_lev, z_lut, kn_z_lut,
                lut_dz, lut_size,
                camera, nbg);
    };

    dz_constant ? launch(ray_tracer_kernel_bw_lw<true>) : launch(ray_tracer_kernel_bw_lw<false>);
}
