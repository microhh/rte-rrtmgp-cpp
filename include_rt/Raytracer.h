#ifndef RAYTRACER_RT_H
#define RAYTRACER_RT_H

#include <memory>
#ifdef USECUDA
#include <curand_kernel.h>
#endif

#include "types.h"
#include "Optical_props_rt.h"
#include "raytracer_definitions.h"

// Forward declarations.
template<typename, int> class Array_gpu;
class Optical_props_rt;
class Optical_props_arry_rt;

#ifdef USECUDA
class Raytracer
{
    public:
        Raytracer();

        void trace_rays(
                const int qrng_gpt_offset,
                const bool switch_independent_column,
                const bool switch_tica,
                const Int photons_per_pixel,
                const Raytracer_definitions::Vector<int> grid_cells,
                const Raytracer_definitions::Vector<Float> grid_d,
                const Raytracer_definitions::Vector<int> kn_grid,
                const Array_gpu<Float,2>& mie_cdf,
                const Array_gpu<Float,3>& mie_ang,
                const Array_gpu<Float,2>& tau_total,
                const Array_gpu<Float,2>& ssa_total,
                const Array_gpu<Float,2>& tau_cloud,
                const Array_gpu<Float,2>& ssa_cloud,
                const Array_gpu<Float,2>& asy_cloud,
                const Array_gpu<Float,2>& tau_aeros,
                const Array_gpu<Float,2>& ssa_aeros,
                const Array_gpu<Float,2>& asy_aeros,
                const Array_gpu<Float,2>& r_eff,
                const Array_gpu<Float,2>& surface_albedo,
                const Float zenith_angle,
                const Float azimuth_angle,
                const Float tod_inc_direct,
                const Float tod_inc_diffuse,
                Array_gpu<Float,2>& flux_tod_dn,
                Array_gpu<Float,2>& flux_tod_up,
                Array_gpu<Float,2>& flux_sfc_dir,
                Array_gpu<Float,2>& flux_sfc_dif,
                Array_gpu<Float,2>& flux_sfc_up,
                Array_gpu<Float,3>& flux_abs_dir,
                Array_gpu<Float,3>& flux_abs_dif);

    private:
        curandDirectionVectors32_t* qrng_vectors_gpu;
        unsigned int* qrng_constants_gpu;
};
#endif

#endif
