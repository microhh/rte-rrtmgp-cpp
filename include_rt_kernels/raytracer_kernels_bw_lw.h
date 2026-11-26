#ifndef RAYTRACER_KERNELS_BW_LW_H
#define RAYTRACER_KERNELS_BW_LW_H

#include <curand_kernel.h>

#include "types.h"
#include "raytracer_definitions.h"
#include "raytracer_functions.h"
#include "raytracer_camera.h"


// CvH TODO THIS SHOULD GO IN A NAMESPACE, CHECK WITH MENNO.

using Raytracer_definitions::Vector;
using Raytracer_definitions::Optics_scat;
using Raytracer_definitions::Grid_knull;
using namespace Raytracer_functions;



#ifdef RTE_USE_SP
constexpr int bw_lw_kernel_block= 128;
constexpr int bw_lw_kernel_grid = 2048;
#else
constexpr int bw_lw_kernel_block = 256;
constexpr int bw_lw_kernel_grid = 256;
#endif



// Host-side launcher for the templated ray_tracer_kernel_bw_lw
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
        const int nbg);

#endif
