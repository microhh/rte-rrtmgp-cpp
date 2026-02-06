//
// cuda version aerosol optics, see src/Aerosol_optics.cpp by Mirjam Tijhuis
//

#ifndef AEROSOL_OPTICS_RT_H
#define AEROSOL_OPTICS_RT_H

#include "array.h"
#include "optical_props_rt.h"
#include "gas_concs.h"
#include "types.h"

using Aerosol_concs_gpu = Gas_concs_gpu;
using Aerosol_concs = Gas_concs;

// Forward declarations.
class Optical_props_rt;

#ifdef USECUDA
class Aerosol_optics_rt : public Optical_props_rt
{
    public:
        Aerosol_optics_rt(
                const Array<Float,2>& band_lims_wvn, const Array<Float,1>& rh_upper,
                const Array<Float,2>& mext_phobic, const Array<Float,2>& ssa_phobic, const Array<Float,2>& g_phobic,
                const Array<Float,3>& mext_philic, const Array<Float,3>& ssa_philic, const Array<Float,3>& g_philic);


        void aerosol_optics(
                const int ibnd,
                Aerosol_concs_gpu& aerosol_concs,
                const Array_gpu<Float,2>& rh, const Array_gpu<Float,2>& plev,
                const bool do_scattering,
                Optical_props_2str_rt& optical_props);

    private:
        // Lookup table coefficients
        Array<Float,2> mext_phobic;
        Array<Float,2> ssa_phobic;
        Array<Float,2> g_phobic;

        Array<Float,3> mext_philic;
        Array<Float,3> ssa_philic;
        Array<Float,3> g_philic;

        Array<Float,1> rh_upper;

        // gpu versions
        Array_gpu<Float,2> mext_phobic_gpu;
        Array_gpu<Float,2> ssa_phobic_gpu;
        Array_gpu<Float,2> g_phobic_gpu;

        Array_gpu<Float,3> mext_philic_gpu;
        Array_gpu<Float,3> ssa_philic_gpu;
        Array_gpu<Float,3> g_philic_gpu;

        Array_gpu<Float,1> rh_upper_gpu;

};
#endif

#endif
