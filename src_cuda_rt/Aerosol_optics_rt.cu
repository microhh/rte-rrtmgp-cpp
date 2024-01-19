//
// cuda version aerosol optics, see src/Aerosol_optics.cpp by Mirjam Tijhuis
//

#include <limits>
#include "Aerosol_optics_rt.h"

namespace
{
    __device__
    int find_rh_class(Float rh, const Float* rh_classes)
    {
        int ihum = 0;
        Float rh_class = rh_classes[ihum];
        while (rh_class < rh)
        {
            ihum += 1;
            rh_class = rh_classes[ihum];
        }

        return ihum;
    }

    __device__
    void add_species_optics(const Float mmr, const Float dpg,
                            const Float mext, const Float ssa, const Float g,
                            Float& tau, Float& taussa, Float& taussag)
    {
        Float local_od = mmr * dpg * mext;
        tau += local_od;
        taussa += local_od * ssa;
        taussag += local_od * ssa * g;
    }


    __global__
    void compute_from_table_kernel(
        const int ncol, const int nlay, const int ibnd, const int nbnd, const int nhum,
        const Float* aermr01, const Float* aermr02, const Float* aermr03,
        const Float* aermr04, const Float* aermr05, const Float* aermr06,
        const Float* aermr07, const Float* aermr08, const Float* aermr09,
        const Float* aermr10, const Float* aermr11,
        const Float* rh, const Float* plev, const Float* rh_classes,
        const Float* mext_phobic, const Float* ssa_phobic, const Float* g_phobic,
        const Float* mext_philic, const Float* ssa_philic, const Float* g_philic,
        Float* tau, Float* taussa, Float* taussag)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( icol < ncol) && ( ilay < nlay) )
        {
            int species_idx;
            Float mmr;
            Float mext;
            Float ssa;
            Float g;

            const int idx = icol + ilay * ncol;
            const int ihum = find_rh_class(rh[idx], rh_classes);

            const Float dpg = abs(plev[idx] - plev[idx + ncol]) / Float(9.81);

            // set to zero
            tau[idx] = Float(0.);
            taussa[idx] = Float(0.);
            taussag[idx] = Float(0.);

            // DU1
            species_idx = ibnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr04[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // DU2
            species_idx = ibnd + 7*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr05[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // DU3
            species_idx = ibnd + 5*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr06[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // BC1
            species_idx = ibnd + 10*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr09[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // BC2
            species_idx = ibnd + 10*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr10[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // SS1
            species_idx = ibnd + ihum*nbnd;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr01[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // SS2
            species_idx = ibnd + ihum*nbnd + 1*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr02[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // SS3
            species_idx = ibnd + ihum*nbnd + 2*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr03[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // SU
            species_idx = ibnd + ihum*nbnd + 4*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr11[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // OM1
            species_idx = ibnd + 9*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr08[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // OM2
            species_idx = ibnd + ihum*nbnd + 3*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr07[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);
        }
    }

    __global__
    void combine_and_store_kernel(const int ncol, const int nlay, const Float tmin,
                  Float* __restrict__ tau, Float* __restrict__ ssa, Float* __restrict__ g,
                  const Float* __restrict__ ltau, const Float* __restrict__ ltaussa, const Float* __restrict__ ltaussag)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (ilay < nlay) )
        {
            const int idx = icol + ilay*ncol;
            tau[idx] = ltau[idx];
            ssa[idx] = ltaussa[idx] / max(tau[idx], tmin);
            g[idx]   = ltaussag[idx] / max(ltaussa[idx], tmin);
        }
    }

    void fill_aerosols_3d(const int ncol, const int nlay, Aerosol_concs_gpu& aerosol_concs)
    {
        for (int i=1; i<=11; ++i)
        {
            std::string name = i<10 ? "aermr0"+std::to_string(i) : "aermr"+std::to_string(i);
            if (aerosol_concs.get_vmr(name).dim(1) == 1)
            {
                aerosol_concs.set_vmr(name, aerosol_concs.get_vmr(name).subset({ {{1, ncol}, {1, nlay}}} ));
            }
        }
    }
}

Aerosol_optics_rt::Aerosol_optics_rt(
        const Array<Float,2>& band_lims_wvn, const Array<Float,1>& rh_upper,
        const Array<Float,2>& mext_phobic, const Array<Float,2>& ssa_phobic, const Array<Float,2>& g_phobic,
        const Array<Float,3>& mext_philic, const Array<Float,3>& ssa_philic, const Array<Float,3>& g_philic) :
        Optical_props_rt(band_lims_wvn)
{
    // Load coefficients.
    this->mext_phobic = mext_phobic;
    this->ssa_phobic = ssa_phobic;
    this->g_phobic = g_phobic;

    this->mext_philic = mext_philic;
    this->ssa_philic = ssa_philic;
    this->g_philic = g_philic;

    this->rh_upper = rh_upper;

    // copy to gpu.
    this->mext_phobic_gpu = this->mext_phobic;
    this->ssa_phobic_gpu = this->ssa_phobic;
    this->g_phobic_gpu = this->g_phobic;

    this->mext_philic_gpu = this->mext_philic;
    this->ssa_philic_gpu = this->ssa_philic;
    this->g_philic_gpu = this->g_philic;

    this->rh_upper_gpu = this->rh_upper;
}



void Aerosol_optics_rt::aerosol_optics(
        const int ibnd,
        Aerosol_concs_gpu& aerosol_concs,
        const Array_gpu<Float,2>& rh, const Array_gpu<Float,2>& plev,
        Optical_props_2str_rt& optical_props)
{
    const int ncol = rh.dim(1);
    const int nlay = rh.dim(2);
    const int nbnd = this->get_nband();
    const int nhum = this->rh_upper.dim(1);

    fill_aerosols_3d(ncol, nlay, aerosol_concs);

    // Temporary arrays for storage.
    Array_gpu<Float,2> ltau    ({ncol, nlay});
    Array_gpu<Float,2> ltaussa ({ncol, nlay});
    Array_gpu<Float,2> ltaussag({ncol, nlay});

    const int block_col = 64;
    const int block_lay = 1;

    const int grid_col  = ncol/block_col + (ncol%block_col > 0);
    const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

    dim3 grid_gpu(grid_col, grid_lay);
    dim3 block_gpu(block_col, block_lay);

    constexpr Float eps = std::numeric_limits<Float>::epsilon();

    compute_from_table_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, ibnd-1, nbnd, nhum,
            aerosol_concs.get_vmr("aermr01").ptr(),
            aerosol_concs.get_vmr("aermr02").ptr(),
            aerosol_concs.get_vmr("aermr03").ptr(),
            aerosol_concs.get_vmr("aermr04").ptr(),
            aerosol_concs.get_vmr("aermr05").ptr(),
            aerosol_concs.get_vmr("aermr06").ptr(),
            aerosol_concs.get_vmr("aermr07").ptr(),
            aerosol_concs.get_vmr("aermr08").ptr(),
            aerosol_concs.get_vmr("aermr09").ptr(),
            aerosol_concs.get_vmr("aermr10").ptr(),
            aerosol_concs.get_vmr("aermr11").ptr(),
            rh.ptr(), plev.ptr(),
            this->rh_upper_gpu.ptr(),
            this->mext_phobic_gpu.ptr(), this->ssa_phobic_gpu.ptr(), this->g_phobic_gpu.ptr(),
            this->mext_philic_gpu.ptr(), this->ssa_philic_gpu.ptr(), this->g_philic_gpu.ptr(),
            ltau.ptr(), ltaussa.ptr(), ltaussag.ptr());

    combine_and_store_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, eps,
            optical_props.get_tau().ptr(), optical_props.get_ssa().ptr(), optical_props.get_g().ptr(),
            ltau.ptr(), ltaussa.ptr(), ltaussag.ptr());



}




