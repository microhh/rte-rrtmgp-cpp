#ifndef RAYTRACER_KERNELS_LW_H
#define RAYTRACER_KERNELS_LW_H

#include "raytracer_functions.h"
#include "raytracer_definitions.h"

using Raytracer_definitions::Vector;
using Raytracer_definitions::Optics_scat;


#ifdef RTE_USE_SP
constexpr int rt_lw_kernel_block = 512;
constexpr int rt_lw_kernel_grid = 1024;
#else
constexpr int rt_lw_kernel_block = 256;
constexpr int rt_lw_kernel_grid = 256;
#endif

constexpr Float k_null_gas_min = Float(1.e-3);

__global__
void ray_tracer_lw_kernel(
        const Float rng_offset,
        const int src_type,
        const bool independent_column,
        const Int photons_to_shoot,
        const float* __restrict__ alias_prob,
        const int* __restrict__ alias_idx,
        const int alias_n,
        const Float* __restrict__ k_null_grid,
        Float* __restrict__ toa_down_count,
        Float* __restrict__ tod_up_count,
        Float* __restrict__ surface_dn_count,
        Float* __restrict__ surface_up_count,
        Float* __restrict__ atmos_count,
        const Float* __restrict__ k_ext,
        const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ surface_emis,
        const Vector<Float> grid_size,
        const Vector<Float> grid_d,
        const Vector<int> grid_cells,
        const Vector<int> kn_grid);
#endif
