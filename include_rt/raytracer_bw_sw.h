#ifndef RAYTRACER_BW_SW_H
#define RAYTRACER_BW_SW_H

#include <memory>
#include "types.h"
#include <curand_kernel.h>
#include "raytracer_kernels_bw_sw.h"
#include "optical_props_rt.h"

// Forward declarations.
template<typename, int> class Array_gpu;
class Optical_props_rt;
class Optical_props_arry_rt;

#ifdef __CUDACC__
class Raytracer_bw_sw
{
    public:
        Raytracer_bw_sw();

        void trace_rays_xyz(
                const int igpt,
                const int photons_per_pixel,
                const int nlay,
                const Vector<int>& grid_cells,
                const Vector<Float>& grid_d,
                const Vector<int>& kn_grid,
                const Array_gpu<Float,1>& z_lev,
                const Array_gpu<Float,1>& kn_z_lev,
                const Array_gpu<int,1>& z_lut,
                const Array_gpu<int,1>& kn_z_lut,
                const Float lut_dz,
                const Float zsize,
                const Array_gpu<Float,3>& mie_cdf,
                const Array_gpu<Float,4>& mie_ang,
                const Array_gpu<Float,4>& mie_phase,
                const Array_gpu<Float,1>& mie_phase_ang,
                const Array_gpu<Float,2>& r_eff,
                const Array_gpu<Float,2>& tau_total,
                const Array_gpu<Float,2>& ssa_total,
                const Array_gpu<Float,2>& tau_cloud,
                const Array_gpu<Float,2>& ssa_cloud,
                const Array_gpu<Float,2>& asy_cloud,
                const Array_gpu<Float,2>& tau_aeros,
                const Array_gpu<Float,2>& ssa_aeros,
                const Array_gpu<Float,2>& asy_aeros,
                const Array_gpu<Float,2>& surface_albedo,
                const Array_gpu<int,1>& surface_scattering_type,
                const Array_gpu<Float,1>& surface_sigma2,
                const Float zenith_angle,
                const Float azimuth_angle,
                const Float src_direct,
                const Float src_diffuse,
                const Float src_factor,
                const Float rayleigh,
                const Array_gpu<Float,2>& col_dry,
                const Array_gpu<Float,2>& vmr_h2o,
                const Camera& camera,
                Array_gpu<Float,2>& flux_camera);

        void trace_rays(
                const int igpt,
                const int photons_per_pixel,
                const int nlay,
                const Vector<int>& grid_cells,
                const Vector<Float>& grid_d,
                const Vector<int>& kn_grid,
                const Array_gpu<Float,1>& z_lev,
                const Array_gpu<Float,1>& kn_z_lev,
                const Array_gpu<int,1>& z_lut,
                const Array_gpu<int,1>& kn_z_lut,
                const Float lut_dz,
                const Float zsize,
                const Array_gpu<Float,3>& mie_cdf,
                const Array_gpu<Float,4>& mie_ang,
                const Array_gpu<Float,4>& mie_phase,
                const Array_gpu<Float,1>& mie_phase_ang,
                const Array_gpu<Float,2>& r_eff,
                const Array_gpu<Float,2>& tau_total,
                const Array_gpu<Float,2>& ssa_total,
                const Array_gpu<Float,2>& tau_cloud,
                const Array_gpu<Float,2>& ssa_cloud,
                const Array_gpu<Float,2>& asy_cloud,
                const Array_gpu<Float,2>& tau_aeros,
                const Array_gpu<Float,2>& ssa_aeros,
                const Array_gpu<Float,2>& asy_aeros,
                const Array_gpu<Float,2>& surface_albedo,
                const Array_gpu<int,1>& surface_scattering_type,
                const Array_gpu<Float,1>& surface_sigma2,
                const Float zenith_angle,
                const Float azimuth_angle,
                const Float src_direct,
                const Float src_diffuse,
                const Camera& camera,
                Array_gpu<Float,2>& flux_camera);

        void add_camera(
                const Camera& camera,
                const Array_gpu<Float,2>& flux_camera,
                Array_gpu<Float,2>& radiance);

        void add_xyz_camera(
                const Camera& camera,
                const Array_gpu<Float,1>& xyz_factor,
                const Array_gpu<Float,2>& flux_camera,
                Array_gpu<Float,3>& XYZ);

        void normalize_xyz_camera(
                const Camera& camera,
                const Float total_source,
                Array_gpu<Float,3>& XYZ);

        void accumulate_clouds(
                const Vector<Float>& grid_d,
                const Vector<int>& grid_cells,
                const Array_gpu<Float,1>& z_lev,
                const Array_gpu<int,1>& z_lut,
                const Float lut_dz,
                const Float zsize,
                const Array_gpu<Float,2>& lwp,
                const Array_gpu<Float,2>& iwp,
                const Array_gpu<Float,2>& tau_cloud,
                const Camera& camera,
                Array_gpu<Float,2>& liwp_cam,
                Array_gpu<Float,2>& tauc_cam,
                Array_gpu<Float,2>& dist_cam,
                Array_gpu<Float,2>& zen_cam);

    private:

};
#endif

#endif
