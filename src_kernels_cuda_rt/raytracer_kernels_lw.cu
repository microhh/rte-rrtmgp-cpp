#include <curand_kernel.h>
#include <iostream>

#include "raytracer_kernels_lw.h"
#include "raytracer_definitions.h"
#include "alias_table.h"

namespace
{
    using namespace Raytracer_functions;

    constexpr Float w_thres = 0.5;

    struct Quasi_random_number_generator_2d
    {
        __device__ Quasi_random_number_generator_2d(
                curandDirectionVectors32_t* vectors, unsigned int* constants, unsigned int offset)
        {
            curand_init(vectors[0], constants[0], offset, &state_x);
            curand_init(vectors[1], constants[1], offset, &state_y);
        }

        __device__ void xy(unsigned int* x, unsigned int* y,
                           const Vector<int>& grid_cells,
                           const Int qrng_grid_x, const Int qrng_grid_y,
                           Int& photons_shot)
        {
            *x = curand(&state_x);
            *y = curand(&state_y);

            while (true)
            {
                const int i = *x / static_cast<unsigned int>((1ULL << 32) / qrng_grid_x);
                const int j = *y / static_cast<unsigned int>((1ULL << 32) / qrng_grid_y);

                ++photons_shot;
                if (i < grid_cells.x && j < grid_cells.y)
                {
                    return;
                }
                else
                {
                    *x = curand(&state_x);
                    *y = curand(&state_y);
                }
            }
        }

        curandStateScrambledSobol32_t state_x;
        curandStateScrambledSobol32_t state_y;
    };

    __device__

    inline int find_source_index(const Float* weights, int n, const Float r)
    {
        int left = 0;
        int right = n;

        while (left < right)
        {
            int mid = left + (right - left) / 2;
            if (weights[mid] <= r)
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }

        return min(left, n-1);
    }

    __device__
    inline void write_emission(
            Photon& photon,
            const int src_type,
            const Float total_absorbed_weight,
            Float* __restrict__ const toa_down_count,
            Float* __restrict__ const surface_up_count,
            Float* __restrict__ const atmos_count)
    {
        if (src_type == 0)
        {
            atomicAdd(&atmos_count[photon.starting_idx], Float(-1.)*total_absorbed_weight);
        }
        if (src_type == 1)
        {
            atomicAdd(&surface_up_count[photon.starting_idx], total_absorbed_weight);
        }
        if (src_type == 2)
        {
            atomicAdd(&toa_down_count[photon.starting_idx], total_absorbed_weight);
        }
    }

    __device__
    inline void reset_photon(
            Photon& photon, const int src_type,
            Int& photons_shot, const Int photons_to_shoot,
            const Float* __restrict__ alias_prob,
            const int* __restrict__ alias_idx,
            const int alias_n,
            Random_number_generator<Float>& rng,
            const Vector<Float> grid_size,
            const Vector<Float> grid_d,
            const Vector<int> grid_cells,
            Float* __restrict__ const toa_down_count,
            Float* __restrict__ const surface_up_count,
            Float* __restrict__ const atmos_count,
            Float& photon_weight,
            Float& total_absorbed_weight)
    {
        ++photons_shot;
        if (photons_shot < photons_to_shoot)
        {
            Float mu, azi;

            if (src_type == 0)
            {
                const int idx = sample_alias_table(
                        alias_prob, alias_idx, alias_n,
                        Float(rng()), Float(rng()));

                const int i = (idx%(grid_cells.x * grid_cells.y)) % grid_cells.x ;
                const int j = (idx%(grid_cells.x * grid_cells.y)) / grid_cells.x ;
                const int k =  idx / (grid_cells.x * grid_cells.y) ;

                const int ij = i + j * grid_cells.x;

                photon.position.x = (i + rng()) * grid_d.x;
                photon.position.y = (j + rng()) * grid_d.y;
                photon.position.z = (k + rng()) * grid_d.z;

                mu = rng()*Float(2.) - Float(1.);
                azi = Float(2.*M_PI)*rng();

                const int ijk = ij + k*grid_cells.x*grid_cells.y;
                photon.starting_idx = ijk;
            }
            if (src_type == 1)
            {
                const int idx = sample_alias_table(
                        alias_prob, alias_idx, alias_n,
                        Float(rng()), Float(rng()));

                const int i = idx % grid_cells.x ;
                const int j = idx / grid_cells.x ;

                const int ij = i + j * grid_cells.x;

                photon.position.x = (i + rng()) * grid_d.x;
                photon.position.y = (j + rng()) * grid_d.y;
                photon.position.z = Float(0.);

                mu = sqrt(rng());
                azi = Float(2.*M_PI)*rng();

                photon.starting_idx = ij;
            }
            if (src_type == 2)
            {
                const int idx = sample_alias_table(
                        alias_prob, alias_idx, alias_n,
                        Float(rng()), Float(rng()));

                const int i = idx % grid_cells.x ;
                const int j = idx / grid_cells.x ;

                const int ij = i + j * grid_cells.x;

                photon.position.x = (i + rng()) * grid_d.x;
                photon.position.y = (j + rng()) * grid_d.y;
                photon.position.z = grid_size.z - Float_epsilon;

                mu = Float(-1.)*sqrt(rng());
                azi = Float(2.*M_PI)*rng();

                photon.starting_idx = ij;
            }

            const Float s = sqrt(Float(1.) - mu*mu + Float_epsilon);
            photon.direction.x = s*sin(azi);
            photon.direction.y = s*cos(azi);
            photon.direction.z = mu;

            photon_weight = Float(1.);
            total_absorbed_weight = Float(0.);

        }
    }

}


__global__
void ray_tracer_lw_kernel(
        const Float rng_offset,
        const int src_type,
        const bool independent_column,
        const Int photons_to_shoot,
        const Float* __restrict__ alias_prob,
        const int* __restrict__ alias_idx,
        const int alias_n,
        const Float* __restrict__ k_null_grid,
        Float* __restrict__ toa_down_count,
        Float* __restrict__ tod_up_count,
        Float* __restrict__ surface_down_count,
        Float* __restrict__ surface_up_count,
        Float* __restrict__ atmos_count,
        const Float* __restrict__ k_ext,
        const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ surface_emis,
        const Vector<Float> grid_size,
        const Vector<Float> grid_d,
        const Vector<int> grid_cells,
        const Vector<int> kn_grid)
{
    const Vector<Float> kn_grid_d = grid_size / kn_grid;

    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    Photon photon;

    // todo, different random seed per g-point
    Random_number_generator<Float> rng(n + rng_offset);

    const Float s_min = max(grid_size.z, max(grid_size.y, grid_size.x)) * Float_epsilon;

    // Set up the initial photons.
    Int photons_shot = Atomic_reduce_const;
    Float photon_weight;
    Float total_absorbed_weight;

    reset_photon(
            photon, src_type,
            photons_shot, photons_to_shoot,
            alias_prob, alias_idx, alias_n, rng,
            grid_size, grid_d, grid_cells,
            toa_down_count, surface_up_count, atmos_count,
            photon_weight, total_absorbed_weight);

    Float tau = Float(0.);
    Float d_max = Float(0.);
    Float k_ext_null;
    bool transition = false;
    int i_n,j_n, k_n;

    while (photons_shot < photons_to_shoot)
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
            d_max = independent_column ? sz : min(sx, min(sy, sz));
            const int ijk_n = i_n + j_n*kn_grid.x + k_n*kn_grid.x*kn_grid.y;
            k_ext_null = k_null_grid[ijk_n];
        }

        if (!transition)
        {
            tau = sample_tau(rng());
        }
        transition = false;
        const Float dn = max(Float_epsilon, tau / k_ext_null);

        if (dn >= d_max)
        {
            if (!independent_column)
            {
                const Float dx = photon.direction.x * (s_min + d_max);
                const Float dy = photon.direction.y * (s_min + d_max);

                photon.position.x += dx;
                photon.position.y += dy;
            }

            const Float dz = photon.direction.z * (s_min + d_max);
            photon.position.z += dz;

            // surface hit
            if (photon.position.z < Float_epsilon)
            {
                photon.position.z = Float_epsilon;
                const int i = float_to_int(photon.position.x, grid_d.x, grid_cells.x);
                const int j = float_to_int(photon.position.y, grid_d.y, grid_cells.y);
                const int ij = i + j*grid_cells.x;
                d_max = Float(0.);

                #ifndef NDEBUG
                if (ij < 0 || ij >=grid_cells.x*grid_cells.y)
                {
                    printf("outofbounds 1 \n");
                }
                #endif

                // // Add surface irradiance
                write_photon_out(&surface_down_count[ij], photon_weight);

                // Update weights and add upward surface flux
                const Float local_albedo = Float(1.) - surface_emis[ij];

                total_absorbed_weight += (1-local_albedo)*photon_weight;

                photon_weight *= local_albedo;
                write_photon_out(&surface_up_count[ij], photon_weight);


                if (photon_weight < w_thres)
                {
                    if (rng() >  photon_weight)
                    {
                        write_emission(photon, src_type, total_absorbed_weight, toa_down_count, surface_up_count, atmos_count);

                        reset_photon(
                             photon, src_type,
                             photons_shot, photons_to_shoot,
                             alias_prob, alias_idx, alias_n, rng,
                             grid_size, grid_d, grid_cells,
                             toa_down_count, surface_up_count, atmos_count,
                             photon_weight, total_absorbed_weight);
                    }
                    else
                    {
                        photon_weight = Float(1.0);
                    }

                }
                // only with nonzero weight continue ray tracing, else start new ray
                if (photon_weight > Float(0.))
                {
                    const Float mu_surface = sqrt(rng());
                    const Float azimuth_surface = Float(2.*M_PI)*rng();
                    const Float sin_theta = sqrt(Float(1.) - mu_surface*mu_surface + Float_epsilon);
                    photon.direction.x = sin_theta*sin(azimuth_surface);
                    photon.direction.y = sin_theta*cos(azimuth_surface);
                    photon.direction.z = mu_surface;
                }
                else
                {
                    write_emission(photon, src_type, total_absorbed_weight, toa_down_count, surface_up_count, atmos_count);

                    reset_photon(
                         photon, src_type,
                         photons_shot, photons_to_shoot,
                         alias_prob, alias_idx, alias_n, rng,
                         grid_size, grid_d, grid_cells,
                         toa_down_count, surface_up_count, atmos_count,
                         photon_weight, total_absorbed_weight);
                    printf ("uh oh, this should not happend \n");
                }
            }

            // TOD exit
            else if (photon.position.z >= grid_size.z)
            {
                d_max = Float(0.);

                const int i = float_to_int(photon.position.x, grid_d.x, grid_cells.x);
                const int j = float_to_int(photon.position.y, grid_d.y, grid_cells.y);
                const int ij = i + j*grid_cells.x;

                #ifndef NDEBUG
                if (ij < 0 || ij >=grid_cells.x*grid_cells.y) printf("Out of bounds at TOD \n");
                #endif

                write_photon_out(&tod_up_count[ij], photon_weight);

                total_absorbed_weight += photon_weight;

                write_emission(photon, src_type, total_absorbed_weight, toa_down_count, surface_up_count, atmos_count);

                reset_photon(
                       photon, src_type,
                       photons_shot, photons_to_shoot,
                       alias_prob, alias_idx, alias_n, rng,
                       grid_size, grid_d, grid_cells,
                       toa_down_count, surface_up_count, atmos_count,
                       photon_weight, total_absorbed_weight);
            }
            // regular cell crossing: adjust tau and apply periodic BC
            else
            {
                photon.position.z += photon.direction.z>0 ? s_min : -s_min;
                if (!independent_column)
                {
                    photon.position.x += photon.direction.x>0 ? s_min : -s_min;
                    photon.position.y += photon.direction.y>0 ? s_min : -s_min;

                    // Cyclic boundary condition in x.
                    photon.position.x = fmod(photon.position.x, grid_size.x);
                    if (photon.position.x < Float(0.))
                        photon.position.x += grid_size.x;

                    // Cyclic boundary condition in y.
                    photon.position.y = fmod(photon.position.y, grid_size.y);
                    if (photon.position.y < Float(0.))
                        photon.position.y += grid_size.y;
                }
                tau -= d_max * k_ext_null;
                d_max = Float(0.);
                transition = true;
            }
        }
        else
        {
            Float dz = photon.direction.z * dn;
            photon.position.z = (dz > 0) ? min(photon.position.z + dz, (k_n+1) * kn_grid_d.z - s_min) : max(photon.position.z + dz, (k_n) * kn_grid_d.z + s_min);

            if (!independent_column)
            {
                Float dx = photon.direction.x * dn;
                Float dy = photon.direction.y * dn;

                photon.position.x = (dx > 0) ? min(photon.position.x + dx, (i_n+1) * kn_grid_d.x - s_min) : max(photon.position.x + dx, (i_n) * kn_grid_d.x + s_min);
                photon.position.y = (dy > 0) ? min(photon.position.y + dy, (j_n+1) * kn_grid_d.y - s_min) : max(photon.position.y + dy, (j_n) * kn_grid_d.y + s_min);
            }

            // Calculate the 3D index.
            const int i = float_to_int(photon.position.x, grid_d.x, grid_cells.x);
            const int j = float_to_int(photon.position.y, grid_d.y, grid_cells.y);
            const int k = float_to_int(photon.position.z, grid_d.z, grid_cells.z);
            const int ijk = i + j*grid_cells.x + k*grid_cells.x*grid_cells.y;

            // Compute probability not being absorbed and store weighted absorption probability
            const Float k_sca_tot = scat_asy[ijk].k_sca_gas + scat_asy[ijk].k_sca_cld + scat_asy[ijk].k_sca_aer;
            const Float ssa_tot = k_sca_tot / k_ext[ijk];

            const Float f_no_abs = max(Float(0.), Float(1.) - (Float(1.) - ssa_tot) * (k_ext[ijk]/k_ext_null));

            #ifndef NDEBUG
            if (ijk < 0 || ijk >= grid_cells.x*grid_cells.y*grid_cells.z) printf("Out of Bounds at Heating Rates %d %d %d %f %f %f  \n",i,j,k,photon.position.x,photon.position.y, photon.position.z);
            #endif

            write_photon_out(&atmos_count[ijk], photon_weight*(1-f_no_abs));

            total_absorbed_weight += photon_weight*(1-f_no_abs);

            // Update weights (see Iwabuchi 2006: https://doi.org/10.1175/JAS3755.1)
            photon_weight *= f_no_abs;
            if (photon_weight < w_thres)
                photon_weight = (rng() > photon_weight) ? Float(0.) : Float(1.);

            // only with nonzero weight continue ray tracing, else start new ray
            if (photon_weight > Float(0.))
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

                    // 0 (gas): rayleigh, 1 (cloud): mie if mie_table_size>0 else HG, 2 (aerosols) HG
                    const Float cos_scat = scatter_type == 0 ? rayleigh(rng()) : henyey(g, rng());
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
                }
            }
            else
            {
                d_max = Float(0.);
                write_emission(photon, src_type, total_absorbed_weight, toa_down_count, surface_up_count, atmos_count);

                reset_photon(
                       photon, src_type,
                       photons_shot, photons_to_shoot,
                       alias_prob, alias_idx, alias_n, rng,
                       grid_size, grid_d, grid_cells,
                       toa_down_count, surface_up_count, atmos_count,
                       photon_weight, total_absorbed_weight);
            }
        }
    }
}
