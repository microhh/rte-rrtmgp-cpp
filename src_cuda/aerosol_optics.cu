//
// cuda version aerosol optics, see src/Aerosol_optics.cpp by Mirjam Tijhuis
//

#include <limits>
#include "aerosol_optics.h"

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
        const int ncol, const int nlay, const int nbnd, const int nhum,
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
        const int ibnd = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( icol < ncol) && ( ilay < nlay) && ( ibnd < nbnd) )
        {
            int species_idx;
            Float mmr;
            Float mext;
            Float ssa;
            Float g;

            const int idx_3d = icol + ilay * ncol + ibnd * nlay * ncol;
            const int idx_2d = icol + ilay * ncol;
            const int ihum = find_rh_class(rh[idx_2d], rh_classes);

            const Float dpg = abs(plev[idx_2d] - plev[idx_2d + ncol]) / Float(9.81);

            // set to zero
            tau[idx_3d] = Float(0.);
            taussa[idx_3d] = Float(0.);
            taussag[idx_3d] = Float(0.);

            // DU1
            species_idx = ibnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr04[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);

            // DU2
            species_idx = ibnd + 7*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr05[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);

            // DU3
            species_idx = ibnd + 5*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr06[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);

            // BC1
            species_idx = ibnd + 10*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr09[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);

            // BC2
            species_idx = ibnd + 10*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr10[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);

            // SS1
            species_idx = ibnd + ihum*nbnd;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr01[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);

            // SS2
            species_idx = ibnd + ihum*nbnd + 1*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr02[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);

            // SS3
            species_idx = ibnd + ihum*nbnd + 2*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr03[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);

            // SU
            species_idx = ibnd + ihum*nbnd + 4*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr11[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);

            // OM1
            species_idx = ibnd + 9*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr08[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);

            // OM2
            species_idx = ibnd + ihum*nbnd + 3*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr07[idx_2d];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx_3d], taussa[idx_3d], taussag[idx_3d]);
        }
    }

    __global__
    void combine_and_store_kernel(const int ncol, const int nlay, const int nbnd, const Float tmin,
                  Float* __restrict__ tau, Float* __restrict__ ssa, Float* __restrict__ g,
                  const Float* __restrict__ ltau, const Float* __restrict__ ltaussa, const Float* __restrict__ ltaussag)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int ibnd = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (ibnd < nbnd) )
        {
            const int idx = icol + ilay*ncol + ibnd*nlay*ncol;
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
                aerosol_concs.set_vmr(name, aerosol_concs.get_vmr(name).subset({ {{1, ncol}, {1, nlay}}} ));
        }
    }
}

Aerosol_optics_gpu::Aerosol_optics_gpu(
        const Array<Float,2>& band_lims_wvn, const Array<Float,1>& rh_upper,
        const Array<Float,2>& mext_phobic, const Array<Float,2>& ssa_phobic, const Array<Float,2>& g_phobic,
        const Array<Float,3>& mext_philic, const Array<Float,3>& ssa_philic, const Array<Float,3>& g_philic) :
        Optical_props_gpu(band_lims_wvn)
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



void Aerosol_optics_gpu::aerosol_optics(
        Aerosol_concs_gpu& aerosol_concs,
        const Array_gpu<Float,2>& rh, const Array_gpu<Float,2>& plev,
        Optical_props_2str_gpu& optical_props)
{
    const int ncol = rh.dim(1);
    const int nlay = rh.dim(2);
    const int nbnd = this->get_nband();
    const int nhum = this->rh_upper.dim(1);

    fill_aerosols_3d(ncol, nlay, aerosol_concs);
    
    // Temporary arrays for storage.
    Array_gpu<Float,3> ltau    ({ncol, nlay, nbnd});
    Array_gpu<Float,3> ltaussa ({ncol, nlay, nbnd});
    Array_gpu<Float,3> ltaussag({ncol, nlay, nbnd});

    const int block_col = 64;
    const int block_lay = 4;
    const int block_bnd = 1;

    const int grid_col  = ncol/block_col + (ncol%block_col > 0);
    const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
    const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);

    dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
    dim3 block_gpu(block_col, block_lay, block_bnd);

    constexpr Float eps = std::numeric_limits<Float>::epsilon();
    compute_from_table_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, nbnd, nhum,
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
            ncol, nlay, nbnd, eps,
            optical_props.get_tau().ptr(), optical_props.get_ssa().ptr(), optical_props.get_g().ptr(),
            ltau.ptr(), ltaussa.ptr(), ltaussag.ptr());


}




