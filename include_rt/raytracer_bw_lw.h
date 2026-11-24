#ifndef RAYTRACER_BW_LW_H
#define RAYTRACER_BW_LW_H

#include <memory>
#include "types.h"
#include <curand_kernel.h>
#include "raytracer_kernels_bw_lw.h"
#include "optical_props_rt.h"

// Forward declarations.
template<typename, int> class Array_gpu;
class Optical_props_rt;
class Optical_props_arry_rt;

#ifdef __CUDACC__
class Raytracer_bw_lw
{
    public:
        Raytracer_bw_lw();

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
                const Array_gpu<Float,2>& tau_total,
                const Array_gpu<Float,2>& ssa_total,
                const Array_gpu<Float,2>& tau_cloud,
                const Array_gpu<Float,2>& ssa_cloud,
                const Array_gpu<Float,2>& asy_cloud,
                const Array_gpu<Float,2>& tau_aeros,
                const Array_gpu<Float,2>& ssa_aeros,
                const Array_gpu<Float,2>& asy_aeros,
                const Array_gpu<Float,2>& lay_source,
                const Array_gpu<Float,1>& sfc_source,
                const Array_gpu<Float,2>& emis_sfc,
                const Float tod_inc_flux,
                const Camera& camera,
                Array_gpu<Float,2>& flux_camera);

        void add_camera(
                const Camera& camera,
                const Array_gpu<Float,2>& flux_camera,
                Array_gpu<Float,2>& radiance);
    private:

};
#endif

#endif
