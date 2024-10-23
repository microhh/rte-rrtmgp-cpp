#include <curand_kernel.h>
#include <iomanip>
#include <iostream>

#include "raytracer_kernels_bw.h"
#include "raytracer_definitions.h"


namespace
{
    using namespace Raytracer_functions;

    constexpr Float w_thres = 0.5;

    enum class Phase_kind {Lambertian, Specular, Rayleigh, HG, Mie};

    template<typename T> __device__
    Vector<T> specular(const Vector<T> dir_in,const Vector<T> dir_n)
    {
        return dir_in - 2*dir_n*dot(dir_n,dir_in);
    }

    __device__
    Float lambertian_phase()
    {
        return Float(1.)/M_PI;
    }

    __device__
    Float rayleigh_phase(const Float cos_angle)
    {
        return Float(3.)/(Float(16.)*M_PI) * (1+cos_angle*cos_angle);
    }

    __device__
    Float henyey_phase(const Float g, const Float cos_angle)
    {
        const Float denom = max(Float_epsilon, 1 + g*g - 2*g*cos_angle);
        return Float(1.)/(Float(4.)*M_PI) * (1-g*g) / (denom*sqrt(denom));
    }

    __device__
    Float transmission_direct_sun(
            Photon photon,
            const int n,
            Random_number_generator<Float>& rng,
            const Vector<Float>& sun_dir,
            const Grid_knull* __restrict__ k_null_grid,
            const Float* __restrict__ k_ext,
            const Float* __restrict__ bg_tau_cum,
            const Float* __restrict__ z_lev_bg,
            const int bg_idx,
            const Vector<int>& kn_grid,
            const Vector<Float>& kn_grid_d,
            const Vector<Float>& grid_d,
            const Vector<Float>& grid_size,
            const Vector<int>& grid_cells,
            const Float s_min, const Float s_min_bg)

    {
        Float tau;
        Float k_ext_null;
        Float k_ext_min;
        Float d_max = Float(0.);
        Float tau_min = Float(0.);

        bool transition = false;
        int i_n,j_n,k_n;

        while (true)
        {
            if (!transition)
            {
                tau = sample_tau(rng());
            }
            transition = false;

            if (photon.position.z > grid_size.z)
            {
                return exp(Float(-1.) * (tau_min + bg_tau_cum[bg_idx]));
            }
            // main grid (dynamical)
            else
            {
                // distance to nearest boundary of acceleration gid voxel
                if (d_max == Float(0.))
                {
                    i_n = float_to_int(photon.position.x, kn_grid_d.x, kn_grid.x);
                    j_n = float_to_int(photon.position.y, kn_grid_d.y, kn_grid.y);
                    k_n = float_to_int(photon.position.z, kn_grid_d.z, kn_grid.z);
                    const Float sx = abs((sun_dir.x > 0) ? ((i_n+1) * kn_grid_d.x - photon.position.x)/sun_dir.x : (i_n*kn_grid_d.x - photon.position.x)/sun_dir.x);
                    const Float sy = abs((sun_dir.y > 0) ? ((j_n+1) * kn_grid_d.y - photon.position.y)/sun_dir.y : (j_n*kn_grid_d.y - photon.position.y)/sun_dir.y);
                    const Float sz = ((k_n+1) * kn_grid_d.z - photon.position.z)/sun_dir.z;
                    d_max = min(sx, min(sy, sz));
                    const int ijk = i_n + j_n*kn_grid.x + k_n*kn_grid.x*kn_grid.y;

                    // decomposition tracking, following Villefranque et al. 2019: minimum k_ext is used to integrate transmissivity, difference max-min as k_null
                    k_ext_min  = k_null_grid[ijk].k_min;
                    k_ext_null = k_null_grid[ijk].k_max - k_ext_min;
                }

                const Float dn = max(Float_epsilon, tau / k_ext_null);
                if (dn >= d_max)
                {
                    // update position
                    tau_min += k_ext_min * d_max;
                    const Float dx = sun_dir.x * d_max;
                    const Float dy = sun_dir.y * d_max;
                    const Float dz = sun_dir.z * d_max;

                    photon.position.x += dx;
                    photon.position.y += dy;
                    photon.position.z += dz;

                    // TOA exit
                    if (photon.position.z >= grid_size.z - s_min)
                    {
                        photon.position.z = grid_size.z + s_min_bg;
                    }
                    // regular cell crossing: adjust tau and apply periodic BC
                    else
                    {
                        photon.position.x += sun_dir.x>0 ? s_min : -s_min;
                        photon.position.y += sun_dir.y>0 ? s_min : -s_min;
                        photon.position.z += sun_dir.z>0 ? s_min : -s_min;

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
                    // hit event: update event and evaluuate change the hit is a null collision
                    tau_min += k_ext_min * dn;
                    const Float dx = sun_dir.x * dn;
                    const Float dy = sun_dir.y * dn;
                    const Float dz = sun_dir.z * dn;

                    photon.position.x = (dx > 0) ? min(photon.position.x + dx, (i_n+1) * kn_grid_d.x - s_min) : max(photon.position.x + dx, (i_n) * kn_grid_d.x + s_min);
                    photon.position.y = (dy > 0) ? min(photon.position.y + dy, (j_n+1) * kn_grid_d.y - s_min) : max(photon.position.y + dy, (j_n) * kn_grid_d.y + s_min);
                    photon.position.z = (dz > 0) ? min(photon.position.z + dz, (k_n+1) * kn_grid_d.z - s_min) : max(photon.position.z + dz, (k_n) * kn_grid_d.z + s_min);

                    // Calculate the 3D index.
                    const int i = float_to_int(photon.position.x, grid_d.x, grid_cells.x);
                    const int j = float_to_int(photon.position.y, grid_d.y, grid_cells.y);
                    const int k = float_to_int(photon.position.z, grid_d.z, grid_cells.z);
                    const int ijk = i + j*grid_cells.x + k*grid_cells.x*grid_cells.y;

                    // Handle the action.
                    const Float k_ext_tot = k_ext[ijk] - k_ext_min;

                    // Compute probability not being absorbed and store weighted absorption probability
                    if (rng() < k_ext_tot/k_ext_null) return 0;

                    d_max -= dn;

                }
            }
        }
    }

    __device__
    inline void reset_photon(
            Photon& photon,
            Float* __restrict__ camera_count,
            Float* __restrict__ camera_shot,
            const int ij_cam, const int n,
            Random_number_generator<Float>& rng,
            const Vector<Float>& sun_direction,
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
            const Float* __restrict__ bg_tau_cum,
            const Float* __restrict__ z_lev_bg,
            const Float s_min,
            const Float s_min_bg)
    {
        const Float i = (Float(ij_cam % camera.nx) + rng())/ Float(camera.nx);
        const Float j = (Float(ij_cam / camera.nx) + rng())/ Float(camera.ny);

        if (camera.fisheye)
        {
            const Float photon_zenith = i * Float(.5) * M_PI / camera.f_zoom;
            const Float photon_azimuth = j * Float(2.) * M_PI;
            const Vector<Float> dir_tmp = {sin(photon_zenith) * sin(photon_azimuth), sin(photon_zenith) * cos(photon_azimuth), cos(photon_zenith)};

            photon.direction.x = dot(camera.mx,  dir_tmp);
            photon.direction.y = dot(camera.my,  dir_tmp);
            photon.direction.z = dot(camera.mz,  dir_tmp) * Float(-1);
        }
        else
        {
            // Rectangular camera based on Villefranque et al. 2019
            photon.direction = normalize(camera.cam_width * (Float(2.)*i-Float(1.0)) + camera.cam_height * (Float(2.)*j-Float(1.0)) + camera.cam_depth);
        }

        photon.position = camera.position + s_min;

        photon.kind = Photon_kind::Direct;
        weight = 1;
        bg_idx = 0;

        for (int i=0; i<kbg; ++i)
        {
            if (photon.position.z > z_lev_bg[i]) bg_idx = i;
        }

        if ( (dot(photon.direction, sun_direction) > cos_half_angle) )
        {
            const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                       k_null_grid,k_ext,
                                       bg_tau_cum, z_lev_bg, bg_idx,
                                       kn_grid, kn_grid_d, grid_d,
                                       grid_size, grid_cells,
                                       s_min, s_min_bg);

           atomicAdd(&camera_count[ij_cam], weight * trans_sun);
        }
        atomicAdd(&camera_shot[ij_cam], Float(1.));
        
    }

    __device__
    inline Float probability_from_sun(
            Photon photon,
            const Vector<Float>& sun_direction,
            const Float sun_solid_angle, const Float g,
            const Float* __restrict__ mie_phase_ang,
            const Float* __restrict__ mie_phase,
            const Float r_eff,
            const int mie_table_size,
            const Vector<Float>& normal,
            const Phase_kind kind)
    {
        const Float cos_angle = dot(photon.direction, sun_direction);
        if (kind == Phase_kind::HG)
        {
            return henyey_phase(g, cos_angle) * sun_solid_angle;
        }
        else if (kind == Phase_kind::Mie)
        {
            // return interpolate_mie_phase_table(mie_phase_ang, mie_phase, max(0.05, acos(cos_angle)), r_eff, mie_table_size) * sun_solid_angle;
            return mie_interpolate_phase_table(mie_phase_ang, mie_phase, acos(cos_angle), r_eff, mie_table_size) * sun_solid_angle;
        }
        else if (kind == Phase_kind::Rayleigh)
        {
            return rayleigh_phase(cos_angle) * sun_solid_angle;
        }
        else if (kind == Phase_kind::Lambertian)
        {
            return lambertian_phase() * sun_solid_angle;
        }
        else if (kind == Phase_kind::Specular)
        {
            return (dot( specular(photon.direction, normal) , sun_direction) > cos_half_angle) ? Float(1.) : Float(0.);
        }
    }
}

__global__
void ray_tracer_kernel_bw(
        const int igpt,
        const int photons_per_pixel,
        const Grid_knull* __restrict__ k_null_grid,
        Float* __restrict__ camera_count,
        Float* __restrict__ camera_shot,
        int* __restrict__ counter,
        const Float* __restrict__ k_ext, const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ k_ext_bg, const Optics_scat* __restrict__ scat_asy_bg,
        const Float* __restrict__ z_lev_bg,
        const Float* __restrict__ r_eff,
        const Float* __restrict__ surface_albedo,
        const Float* __restrict__ land_use_map,
        const Float mu,
        const Vector<Float> grid_size, const Vector<Float> grid_d,
        const Vector<int> grid_cells, const Vector<int> kn_grid,
        const Vector<Float> sun_direction, const Camera camera,
        const int kbg,
        const Float* __restrict__ mie_cdf,
        const Float* __restrict__ mie_ang,
        const Float* __restrict__ mie_phase,
        const Float* __restrict__ mie_phase_ang,
        const int mie_table_size)
{
    extern __shared__ Float shared_arrays[];
    Float* mie_cdf_shared = &shared_arrays[0];
    Float* mie_phase_ang_shared = &shared_arrays[mie_table_size];
    Float* bg_tau_cum = &shared_arrays[2*mie_table_size];
    if (threadIdx.x==0)
    {
        if (mie_table_size > 0)
        {
            for (int mie_i=0; mie_i<mie_table_size; ++mie_i)
            {
                mie_cdf_shared[mie_i] = mie_cdf[mie_i];
                mie_phase_ang_shared[mie_i] = mie_phase_ang[mie_i];
            }
        }

        Float bg_tau = Float(0.);
        for (int k=kbg-1; k >= 0; --k)
        {
            bg_tau += k_ext_bg[k] * abs(z_lev_bg[k+1]-z_lev_bg[k]) / mu;
            bg_tau_cum[k] = bg_tau;
        }
    }

    __syncthreads();
    
    Vector<Float> surface_normal = {0, 0, 1};
    
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    
    const Float bg_transmissivity = exp(-bg_tau_cum[0]);

    const Vector<Float> kn_grid_d = grid_size / kn_grid;
    const Float z_top = z_lev_bg[kbg];

    Random_number_generator<Float> rng(n+bw_kernel_block*bw_kernel_grid*igpt);

    const Float s_min = max(max(grid_size.z, grid_size.x), grid_size.y) * Float_epsilon;
    const Float s_min_bg = max(max(grid_size.x, grid_size.y), z_top) * Float_epsilon;

    while (counter[0] < camera.nx*camera.ny*photons_per_pixel)
    {
        const int count = atomicAdd(&counter[0], 1);
        const int ij_cam = count / photons_per_pixel;

        if (ij_cam >= camera.nx*camera.ny)
            return;

        Float weight;
        int bg_idx;

        Photon photon;

        reset_photon(
                photon, camera_count, camera_shot,
                ij_cam, n, rng, sun_direction,
                k_null_grid, k_ext,
                kn_grid, kn_grid_d, grid_d,
                grid_size, grid_cells,
                weight, bg_idx,
                camera,
                kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);

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

                        // direct contribution
                        const Phase_kind kind = (scatter_type==0) ? Phase_kind::Rayleigh : Phase_kind::HG;
                        const Float p_sun = probability_from_sun(photon, sun_direction, sun_solid_angle, g, mie_phase_ang_shared, mie_phase, Float(0.), 0, surface_normal, kind);
                        const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                                    k_null_grid,k_ext,
                                                    bg_tau_cum, z_lev_bg, bg_idx,
                                                    kn_grid, kn_grid_d, grid_d,
                                                    grid_size, grid_cells,
                                                    s_min, s_min_bg);
                        atomicAdd(&camera_count[ij_cam], weight * p_sun * trans_sun);


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
                    k_n = float_to_int(photon.position.z, kn_grid_d.z, kn_grid.z);
                    const Float sx = abs((photon.direction.x > 0) ? ((i_n+1) * kn_grid_d.x - photon.position.x)/photon.direction.x : (i_n*kn_grid_d.x - photon.position.x)/photon.direction.x);
                    const Float sy = abs((photon.direction.y > 0) ? ((j_n+1) * kn_grid_d.y - photon.position.y)/photon.direction.y : (j_n*kn_grid_d.y - photon.position.y)/photon.direction.y);
                    const Float sz = abs((photon.direction.z > 0) ? ((k_n+1) * kn_grid_d.z - photon.position.z)/photon.direction.z : (k_n*kn_grid_d.z - photon.position.z)/photon.direction.z);
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

                        // Update weights and add upward surface flux
                        const Float local_albedo =  surface_albedo[ij];
                        weight *= local_albedo;

                        // only specular reflection for water surfaces and direct radiation, otherwise keep using Lambertian to diffuse a bit
                        const Phase_kind surface_kind = (land_use_map[ij] == 0) ? ( (photon.kind == Photon_kind::Direct) ? Phase_kind::Specular : Phase_kind::Lambertian)
                                                                                : Phase_kind::Lambertian;

                        // SUN SCATTERING GOES HERE
                        const Float p_sun = probability_from_sun(photon, sun_direction, sun_solid_angle, Float(0.),  mie_phase_ang_shared, mie_phase, Float(0.), 0,
                                                                 surface_normal, surface_kind);
                        const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                                    k_null_grid,k_ext,
                                                    bg_tau_cum, z_lev_bg, bg_idx,
                                                    kn_grid, kn_grid_d, grid_d,
                                                    grid_size, grid_cells,
                                                    s_min, s_min_bg);
                        atomicAdd(&camera_count[ij_cam], weight * p_sun * trans_sun);

                        if (weight < w_thres)
                            weight = (rng() > weight) ? Float(0.) : Float(1.);

                        // only with nonzero weight continue ray tracing, else start new ray
                        if (weight > Float(0.))
                        {
                            if (surface_kind == Phase_kind::Lambertian)
                            {
                                const Float mu_surface = sqrt(rng());
                                const Float azimuth_surface = Float(2.*M_PI)*rng();

                                photon.direction.x = mu_surface*sin(azimuth_surface);
                                photon.direction.y = mu_surface*cos(azimuth_surface);
                                photon.direction.z = sqrt(Float(1.) - mu_surface*mu_surface + Float_epsilon);
                                photon.kind = Photon_kind::Diffuse;
                            }
                            else if (surface_kind == Phase_kind::Specular)
                            {
                                photon.direction = specular(photon.direction,surface_normal);
                            }
                        }
                        else
                        {
                            photon_alive = false;
                        }
                    }

                    // TOD exit
                    else if (photon.position.z >= grid_size.z)
                    {
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
                    photon.position.z = (dz > 0) ? min(photon.position.z + dz, (k_n+1) * kn_grid_d.z - s_min) : max(photon.position.z + dz, (k_n) * kn_grid_d.z + s_min);

                    // Calculate the 3D index.
                    const int i = float_to_int(photon.position.x, grid_d.x, grid_cells.x);
                    const int j = float_to_int(photon.position.y, grid_d.y, grid_cells.y);
                    const int k = float_to_int(photon.position.z, grid_d.z, grid_cells.z);
                    const int ijk = i + j*grid_cells.x + k*grid_cells.x*grid_cells.y;


                    // Compute probability not being absorbed and store weighted absorption probability
                    const Float k_sca_tot = scat_asy[ijk].k_sca_gas + scat_asy[ijk].k_sca_cld + scat_asy[ijk].k_sca_aer;
                    const Float ssa_tot = k_sca_tot / k_ext[ijk];
                    const Float f_no_abs = Float(1.) - (Float(1.) - ssa_tot) * (k_ext[ijk]/k_ext_null);

                    // Update weights (see Iwabuchi 2006: https://doi.org/10.1175/JAS3755.1)
                    weight *= f_no_abs;

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
                            const Float cos_scat = scatter_type == 0 ? rayleigh(rng()) : // gases -> rayleigh,
                                                                   1 ? ( (mie_table_size > 0) //clouds: Mie or HG
                                                                            ? cos( mie_sample_angle(mie_cdf_shared, mie_ang, rng(), r_eff[ijk], mie_table_size) )
                                                                            :  henyey(g, rng()))
                                                                   : henyey(g, rng()); //aerosols
                            const Float sin_scat = max(Float(0.), sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon));

                            // SUN SCATTERING GOES HERE
                            const Phase_kind kind = scatter_type == 0 ? Phase_kind::Rayleigh :
                                                                    1 ? (mie_table_size > 0)
                                                                        ? Phase_kind::Mie
                                                                        : Phase_kind::HG
                                                                : Phase_kind::HG;
                            const Float p_sun = probability_from_sun(photon, sun_direction, sun_solid_angle, g, mie_phase_ang_shared, mie_phase, r_eff[ijk], mie_table_size,
                                                                     surface_normal, kind);
                            const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                                        k_null_grid,k_ext,
                                                        bg_tau_cum, z_lev_bg, bg_idx,
                                                        kn_grid, kn_grid_d, grid_d,
                                                        grid_size, grid_cells,
                                                        s_min, s_min_bg);
                            atomicAdd(&camera_count[ij_cam], weight * p_sun * trans_sun);

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
    __global__
    void accumulate_clouds_kernel(
            const Float* __restrict__ lwp, 
            const Float* __restrict__ iwp, 
            const Float* __restrict__ tau_cloud, 
            const Vector<Float> grid_d,
            const Vector<Float> grid_size,
            const Vector<int> grid_cells,
            Float* __restrict__ liwp_cam, 
            Float* __restrict__ tauc_cam, 
            Float* __restrict__ dist_cam, 
            Float* __restrict__ zen_cam, 
            const Camera camera)
    {
        const int pix = blockDim.x * blockIdx.x + threadIdx.x;
        const Float s_eps = max(max(grid_size.z, grid_size.x), grid_size.y) * Float_epsilon;
        Vector<Float> direction; 
        Vector<Float> position;
    
        if (pix < camera.nx * camera.ny)
        {
            Float liwp_sum = 0;
            Float tauc_sum = 0;
            Float dist = 0;
            bool reached_cloud = false;

            //const int pix = n * pixels_per_thread + ipix;
            const Float i = (Float(pix % camera.nx) + Float(0.5))/ Float(camera.nx);
            const Float j = (Float(pix / camera.nx) + Float(0.5))/ Float(camera.ny);

            position = camera.position + s_eps;

            if (camera.fisheye)
            {
                const Float photon_zenith = i * Float(.5) * M_PI / camera.f_zoom;
                const Float photon_azimuth = j * Float(2.) * M_PI;
                const Vector<Float> dir_tmp = {sin(photon_zenith) * sin(photon_azimuth), sin(photon_zenith) * cos(photon_azimuth), cos(photon_zenith)};

                direction = {dot(camera.mx,  dir_tmp), dot(camera.my,  dir_tmp), dot(camera.mz,  dir_tmp) * Float(-1)};
            }
            else
            {
                direction = normalize(camera.cam_width * (Float(2.)*i-Float(1.0)) + camera.cam_height * (Float(2.)*j-Float(1.0)) + camera.cam_depth);
            }
            
            // first bring photon to top of dynamical domain
            if ((position.z >= (grid_size.z - s_eps)) && (direction.z < Float(0.)))
            {
                const Float s = abs((position.z - grid_size.z)/direction.z);
                position = position + direction * s - s_eps;
                
                // Cyclic boundary condition in x.
                position.x = fmod(position.x, grid_size.x);
                if (position.x < Float(0.))
                    position.x += grid_size.x;

                // Cyclic boundary condition in y.
                position.y = fmod(position.y, grid_size.y);
                if (position.y < Float(0.))
                    position.y += grid_size.y;
            }

            while ((position.z <= grid_size.z - s_eps) && (position.z >= s_eps))
            {
                const int i = float_to_int(position.x, grid_d.x, grid_cells.x);
                const int j = float_to_int(position.y, grid_d.y, grid_cells.y);
                const int k = float_to_int(position.z, grid_d.z, grid_cells.z);
                const int ijk = i + j*grid_cells.x + k*grid_cells.x*grid_cells.y;
                
                const Float sx = abs((direction.x > 0) ? ((i+1) * grid_d.x - position.x)/direction.x : (i*grid_d.x - position.x)/direction.x);
                const Float sy = abs((direction.y > 0) ? ((j+1) * grid_d.y - position.y)/direction.y : (j*grid_d.y - position.y)/direction.y);
                const Float sz = abs((direction.z > 0) ? ((k+1) * grid_d.z - position.z)/direction.z : (k*grid_d.z - position.z)/direction.z);
                const Float s_min = min(sx, min(sy, sz));
                
                liwp_sum += s_min * (lwp[ijk] + iwp[ijk]);
                tauc_sum += s_min * tau_cloud[ijk];
                if (!reached_cloud)
                {
                    dist += s_min;
                    reached_cloud = tau_cloud[ijk] > 0;
                }
                
                position = position + direction * s_min;

                position.x += direction.x >= 0 ? s_eps : -s_eps;
                position.y += direction.y >= 0 ? s_eps : -s_eps;
                position.z += direction.z >= 0 ? s_eps : -s_eps;
                        
                // Cyclic boundary condition in x.
                position.x = fmod(position.x, grid_size.x);
                if (position.x < Float(0.))
                    position.x += grid_size.x;

                // Cyclic boundary condition in y.
                position.y = fmod(position.y, grid_size.y);
                if (position.y < Float(0.))
                    position.y += grid_size.y;

            }
            
            // divide out initial layer thicknes, equivalent to first converting lwp (g/m2) to lwc (g/m3) or optical depth to k_ext(1/m)
            liwp_cam[pix] = liwp_sum / grid_d.z;
            tauc_cam[pix] = tauc_sum / grid_d.z;
            dist_cam[pix] = reached_cloud ? dist : Float(-1) ;
            zen_cam[pix] = acos(direction.z);
        }




    }



