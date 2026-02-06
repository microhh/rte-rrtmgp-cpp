#ifndef RAYTRACER_LW_RT_H
#define RAYTRACER_LW_RT_H

#include <memory>
#ifdef USECUDA
#include <curand_kernel.h>
#endif

#include "types.h"
#include "optical_props_rt.h"
#include "raytracer_definitions.h"

// Forward declarations.
template<typename, int> class Array_gpu;
class Optical_props_rt;
class Optical_props_arry_rt;

#ifdef USECUDA
class Raytracer_lw
{
    public:
        Raytracer_lw();

        void trace_rays(
                const int igpt,
                const bool switch_independent_column,
                const Int n_photons,
                const Raytracer_definitions::Vector<int> grid_cells,
                const Raytracer_definitions::Vector<Float> grid_d,
                const Raytracer_definitions::Vector<int> kn_grid,
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
                Array_gpu<Float,2>& flux_tod_dn,
                Array_gpu<Float,2>& flux_tod_up,
                Array_gpu<Float,2>& flux_sfc_dn,
                Array_gpu<Float,2>& flux_sfc_up,
                Array_gpu<Float,3>& flux_abs);

    private:
//        curandDirectionVectors32_t* qrng_vectors_gpu;
//        unsigned int* qrng_constants_gpu;
};
#endif

#endif
