#ifndef RAYTRACER_KERNELS_H
#define RAYTRACER_KERNELS_H

#include "raytracer_functions.h"
#include "raytracer_definitions.h"

using Raytracer_definitions::Vector;
using Raytracer_definitions::Optics_scat;


#ifdef RTE_USE_SP
constexpr int rt_kernel_block = 512;
constexpr int rt_kernel_grid = 1024;
#else
constexpr int rt_kernel_block = 512;
constexpr int rt_kernel_grid = 256;
#endif

constexpr Float k_null_gas_min = Float(1.e-3);

__global__
void ray_tracer_kernel(
        const Bool independent_column,
        const Bool tica,
        const Int photons_to_shoot,
        const Int qrng_grid_x,
        const Int qrng_grid_y,
        const Int qrng_gpt_offset,
        const Float* __restrict__ k_null_grid,
        Float* __restrict__ toa_down_count,
        Float* __restrict__ tod_up_count,
        Float* __restrict__ surface_down_direct_count,
        Float* __restrict__ surface_down_diffuse_count,
        Float* __restrict__ surface_up_count,
        Float* __restrict__ atmos_direct_count,
        Float* __restrict__ atmos_diffuse_count,
        const Float* __restrict__ k_ext,
        const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ r_eff,
        const Float tod_inc_direct,
        const Float tod_inc_diffuse,
        const Float* __restrict__ surface_albedo,
        const Vector<Float> grid_size,
        const Vector<Float> grid_d,
        const Vector<int> grid_cells,
        const Vector<int> kn_grid,
        const Vector<Float> sun_direction,
        curandDirectionVectors32_t* qrng_vectors, unsigned int* qrng_constants,
        const Float* __restrict__ mie_cdf,
        const Float* __restrict__ mie_ang,
        const int mie_table_size);
#endif
