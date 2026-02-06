//
// Created by Mirjam Tijhuis on 05/08/2022.
//

#ifndef AEROSOL_OPTICS_H
#define AEROSOL_OPTICS_H
#include "array.h"
#include "optical_props.h"
#include "gas_concs.h"
#include "types.h"


// Forward declarations.
class Optical_props;
class Optical_props_gpu;
class Gas_concs;
class Gas_concs_gpu;

using Aerosol_concs = Gas_concs;
using Aerosol_concs_gpu = Gas_concs_gpu;

class Aerosol_optics : public Optical_props
{
    public:
        Aerosol_optics(
                const Array<Float,2>& band_lims_wvn, const Array<Float,1>& rh_upper,
                const Array<Float,2>& mext_phobic, const Array<Float,2>& ssa_phobic, const Array<Float,2>& g_phobic,
                const Array<Float,3>& mext_philic, const Array<Float,3>& ssa_philic, const Array<Float,3>& g_philic);

        void aerosol_optics(
                Aerosol_concs& aerosol_concs,
                const Array<Float,2>& rh, const Array<Float,2>& plev,
                Optical_props_2str& optical_props);

    private:
        // Lookup table coefficients.
        Array<Float,2> mext_phobic;
        Array<Float,2> ssa_phobic;
        Array<Float,2> g_phobic;

        Array<Float,3> mext_philic;
        Array<Float,3> ssa_philic;
        Array<Float,3> g_philic;

        Array<Float,1> rh_upper;

};

#ifdef USECUDA
class Aerosol_optics_gpu : public Optical_props_gpu
{
    public:
        Aerosol_optics_gpu(
                const Array<Float,2>& band_lims_wvn, const Array<Float,1>& rh_upper,
                const Array<Float,2>& mext_phobic, const Array<Float,2>& ssa_phobic, const Array<Float,2>& g_phobic,
                const Array<Float,3>& mext_philic, const Array<Float,3>& ssa_philic, const Array<Float,3>& g_philic);

        void aerosol_optics(
                Aerosol_concs_gpu& aerosol_concs,
                const Array_gpu<Float,2>& rh, const Array_gpu<Float,2>& plev,
                Optical_props_2str_gpu& optical_props);

    private:
        // Lookup table coefficients.
        Array<Float,2> mext_phobic;
        Array<Float,2> ssa_phobic;
        Array<Float,2> g_phobic;

        Array<Float,3> mext_philic;
        Array<Float,3> ssa_philic;
        Array<Float,3> g_philic;

        Array<Float,1> rh_upper;

        Array_gpu<Float,2> mext_phobic_gpu;
        Array_gpu<Float,2> ssa_phobic_gpu;
        Array_gpu<Float,2> g_phobic_gpu;

        Array_gpu<Float,3> mext_philic_gpu;
        Array_gpu<Float,3> ssa_philic_gpu;
        Array_gpu<Float,3> g_philic_gpu;

        Array_gpu<Float,1> rh_upper_gpu;
};

#endif

#endif //AEROSOL_OPTICS_H
