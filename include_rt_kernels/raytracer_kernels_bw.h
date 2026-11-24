#ifndef RAYTRACER_KERNELS_BW_H
#define RAYTRACER_KERNELS_BW_H

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
constexpr int bw_kernel_block= 128;
constexpr int bw_kernel_grid = 2048;
#else
constexpr int bw_kernel_block = 256;
constexpr int bw_kernel_grid = 256;
#endif

// sun has a half angle of .266 degrees
constexpr Float cos_half_angle = Float(0.9999891776066407); // cos(half_angle);
constexpr Float sun_solid_angle = Float(6.799910294339209e-05); // 2.*M_PI*(1-cos_half_angle);
constexpr Float sun_solid_angle_reciprocal = Float(14706.07635563193);
constexpr Float sun_hemisphere_sa_fraction = Float(1.0822393359255502e-05);



// Host-side launcher for the templated ray_tracer_kernel_bw
void launch_ray_tracer_kernel_bw(
        const bool tod_diffuse,
        const bool dz_constant,
        const dim3 grid,
        const dim3 block,
        const size_t shared_mem_bytes,
        const int igpt,
        const Int photons_per_pixel,
        const Grid_knull* __restrict__ k_null_grid,
        Float* __restrict__ camera_count_direct,
        Float* __restrict__ camera_count_diffuse,
        Float* __restrict__ camera_shot,
        Int* __restrict__ counter,
        const Float* __restrict__ k_ext, const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ k_ext_bg, const Optics_scat* __restrict__ scat_asy_bg,
        const Float* __restrict__ z_lev_bg,
        const Float* __restrict__ r_eff,
        const Surface_props* __restrict__ surface_properties,
        const Float mu,
        const Vector<Float> grid_size,
        const Vector<Float> grid_d,
        const Vector<int> grid_cells,
        const Vector<int> kn_grid,
        const Float* __restrict__ z_lev,
        const Float* __restrict__ kn_z_lev,
        const int* __restrict__ z_lut,
        const int* __restrict__ kn_z_lut,
        const Float lut_dz,
        const int lut_size,
        const Vector<Float> sun_direction,
        const Camera camera,
        const int nbg,
        const Float* __restrict__ mie_cdf,
        const Float* __restrict__ mie_ang,
        const Float* __restrict__ mie_phase,
        const Float* __restrict__ mie_phase_ang,
        const int mie_cdf_table_size,
        const int mie_phase_table_size);

__global__
void accumulate_clouds_kernel(
    const Float* __restrict__ lwp,
    const Float* __restrict__ iwp,
    const Float* __restrict__ tau_cloud,
    const Vector<Float> grid_d,
    const Vector<Float> grid_size,
    const Vector<int> grid_cells,
    const Float* __restrict__ z_lev,
    const int* __restrict__ z_lut,
    const Float lut_dz,
    const int lut_size,
    Float* __restrict__ liwp_cam,
    Float* __restrict__ tauc_cam,
    Float* __restrict__ dist_cam,
    Float* __restrict__ zen_cam,
    const Camera camera);

#endif
