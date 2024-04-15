/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/earth-system-radiation/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/earth-system-radiation/rte-rrtmgp-cpp
 *
 * Contact: Chiel van Heerwaarden
 * email: chiel.vanheerwaarden@wur.nl
 *
 * Copyright 2020, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#ifndef RRTMGP_KERNELS_H
#define RRTMGP_KERNELS_H

#include "types.h"


// Kernels of fluxes.
namespace rrtmgp_kernels
{
    extern "C" void rte_sum_broadband(
            int* ncol, int* nlev, int* ngpt,
            Float* spectral_flux, Float* broadband_flux);

    extern "C" void rte_net_broadband_precalc(
            int* ncol, int* nlev,
            Float* broadband_flux_dn, Float* broadband_flux_up,
            Float* broadband_flux_net);

    extern "C" void sum_byband(
            int* ncol, int* nlev, int* ngpt, int* nbnd,
            int* band_lims,
            Float* spectral_flux,
            Float* byband_flux);

    extern "C" void net_byband_precalc(
            int* ncol, int* nlev, int* nbnd,
            Float* byband_flux_dn, Float* byband_flux_up,
            Float* byband_flux_net);

    extern "C" void zero_array_3D(
            int* ni, int* nj, int* nk, Float* array);

    extern "C" void zero_array_4D(
             int* ni, int* nj, int* nk, int* nl, Float* array);

    extern "C" void rrtmgp_interpolation(
                int* ncol, int* nlay,
                int* ngas, int* nflav, int* neta, int* npres, int* ntemp,
                int* flavor,
                Float* press_ref_log,
                Float* temp_ref,
                Float* press_ref_log_delta,
                Float* temp_ref_min,
                Float* temp_ref_delta,
                Float* press_ref_trop_log,
                Float* vmr_ref,
                Float* play,
                Float* tlay,
                Float* col_gas,
                int* jtemp,
                Float* fmajor, Float* fminor,
                Float* col_mix,
                Bool* tropo,
                int* jeta,
                int* jpress);

    extern "C" void rrtmgp_compute_tau_absorption(
            int* ncol, int* nlay, int* nband, int* ngpt,
            int* ngas, int* nflav, int* neta, int* npres, int* ntemp,
            int* nminorlower, int* nminorklower,
            int* nminorupper, int* nminorkupper,
            int* idx_h2o,
            int* gpoint_flavor,
            int* band_lims_gpt,
            Float* kmajor,
            Float* kminor_lower,
            Float* kminor_upper,
            int* minor_limits_gpt_lower,
            int* minor_limits_gpt_upper,
            Bool* minor_scales_with_density_lower,
            Bool* minor_scales_with_density_upper,
            Bool* scale_by_complement_lower,
            Bool* scale_by_complement_upper,
            int* idx_minor_lower,
            int* idx_minor_upper,
            int* idx_minor_scaling_lower,
            int* idx_minor_scaling_upper,
            int* kminor_start_lower,
            int* kminor_start_upper,
            Bool* tropo,
            Float* col_mix, Float* fmajor, Float* fminor,
            Float* play, Float* tlay, Float* col_gas,
            int* jeta, int* jtemp, int* jpress,
            Float* tau);

    extern "C" void reorder_123x321_kernel(
            int* dim1, int* dim2, int* dim3,
            Float* array, Float* array_out);

    extern "C" void combine_and_reorder_2str(
            int* ncol, int* nlay, int* ngpt,
            Float* tau_local, Float* tau_rayleigh,
            Float* tau, Float* ssa, Float* g);

    extern "C" void rrtmgp_compute_Planck_source(
            int* ncol, int* nlay, int* nbnd, int* ngpt,
            int* nflav, int* neta, int* npres, int* ntemp, int* nPlanckTemp,
            Float* tlay, Float* tlev, Float* tsfc, int* sfc_lay,
            Float* fmajor, int* jeta, Bool* tropo, int* jtemp, int* jpress,
            int* gpoint_bands, int* band_lims_gpt, Float* pfracin, Float* temp_ref_min,
            Float* totplnk_delta, Float* totplnk, int* gpoint_flavor,
            Float* sfc_src, Float* lay_src, Float* lev_src, Float* lev_source_dec,
            Float* sfc_src_jac);

    extern "C" void rrtmgp_compute_tau_rayleigh(
            int* ncol, int* nlay, int* nband, int* ngpt,
            int* ngas, int* nflav, int* neta, int* npres, int* ntemp,
            int* gpoint_flavor,
            int* band_lims_gpt,
            Float* krayl,
            int* idx_h2o, Float* col_dry, Float* col_gas,
            Float* fminor, int* eta,
            Bool* tropo, int* jtemp,
            Float* tau_rayleigh);

    /*
    extern "C" void apply_BC_0(
            const int& ncol,
            const int& nlay,
            const int& ngpt,
            const Bool& top_at_1,
            Float* gpt_flux_dn);

    extern "C" void apply_BC_gpt(
            const int& ncol,
            const int& nlay,
            const int& ngpt,
            const Bool& top_at_1,
            const Float* inc_flux,
            Float* gpt_flux_dn);
            */

    // ROBERTS REF: REMOVE COMMENTS.
    // void rte_lw_solver_noscat(
    //         const int& ncol, 
    //         const int& nlay, 
    //         const int& ngpt,
    //         const Bool& top_at_1, 
    //         const int& nmus,
    //         const Float* secants, // (nmus)
    //         const Float* weights, // (nmus)
    //         const Float* tau,        // (ncol,nlay,  ngpt)
    //         const Float* lay_source, // (ncol,nlay,  ngpt)
    //         const Float* lev_source, // (ncol,nlay+1,ngpt)
    //         const Float* sfc_emis,   // (ncol,       ngpt)
    //         const Float* sfc_src,    // (ncol,       ngpt)
    //         const Float* inc_flux,   // (ncol,       ngpt)
    //         Float* flux_up,  // [out]   (ncol,nlay+1,ngpt)
    //         Float* flux_dn,  // [out]   (ncol,nlay+1,ngpt)
    //         const Bool& do_broadband,
    //         Float* broadband_up,
    //                          // [out]   (ncol,nlay+1)
    //         Float* broadband_dn,
    //                          // [out]   (ncol,nlay+1)
    //         const Bool& do_jacobians,
    //         const Float* sfc_src_jac,
    //                                // (ncol,       ngpt)
    //         Float* flux_up_jac,
    //                        // [out]   (ncol,nlay+1,ngpt)
    //         const Bool& do_rescaling,
    //         const Float* ssa,      // (ncol,nlay,  ngpt)
    //         const Float* g);       // (ncol,nlay,  ngpt)


    extern "C" void rte_lw_solver_noscat(
            const int& ncol,
            const int& nlay,
            const int& ngpt,
            const Bool& top_at_1,
            const int& n_quad_angs,
            const Float* secants,
            const Float* gauss_wts_subset,
            const Float* tau,
            const Float* lay_source,
            const Float* lev_source_inc,
            const Float* lev_source_dec,
            const Float* sfc_emis_gpt,
            const Float* sfc_source,
            const Float* inc_flux_diffuse,
            Float* gpt_flux_up,
            Float* gpt_flux_dn,
            const Bool& do_broadband,
            Float* flux_up_loc,
            Float* flux_dn_loc,
            const Bool& do_jacobians,
            const Float* sfc_source_jac,
            Float* gpt_flux_up_jac,
            const Bool& do_rescaling,
            const Float* ssa,
            const Float* g);

    /*
    extern "C" void apply_BC_factor(
            const int& ncol,
            const int& nlay,
            const int& ngpt,
            const Bool& top_at_1,
            const Float* inc_flux,
            const Float* factor,
            Float* flux_dn);
            */

    extern "C" void rte_sw_solver_2stream(
            const int& ncol,
            const int& nlay,
            const int& ngpt,
            const Bool& top_at_1,
            const Float* tau,
            const Float* ssa,
            const Float* g,
            const Float* mu0,
            const Float* sfc_alb_dir_gpt,
            const Float* sfc_alb_dif_gpt,
            const Float* inc_flux_dir,
            Float* gpt_flux_up,
            Float* gpt_flux_dn,
            Float* gpt_flux_dir,
            const Bool& has_dif_bc,
            const Float* inc_flux_dif,
            const Bool& do_broadband,
            Float* flux_up_loc,
            Float* flux_dn_loc,
            Float* flux_dir_loc);

    extern "C" void rte_increment_2stream_by_2stream(
            int* ncol, int* nlev, int* ngpt,
            Float* tau_inout, Float* ssa_inout, Float* g_inout,
            Float* tau_in, Float* ssa_in, Float* g_in);

    extern "C" void rte_increment_1scalar_by_1scalar(
            int* ncol, int* nlev, int* ngpt,
            Float* tau_inout, Float* tau_in);

    extern "C" void rte_inc_2stream_by_2stream_bybnd(
            int* ncol, int* nlev, int* ngpt,
            Float* tau_inout, Float* ssa_inout, Float* g_inout,
            Float* tau_in, Float* ssa_in, Float* g_in,
            int* nbnd, int* band_lims_gpoint);

    extern "C" void rte_inc_1scalar_by_1scalar_bybnd(
            int* ncol, int* nlev, int* ngpt,
            Float* tau_inout, Float* tau_in,
            int* nbnd, int* band_lims_gpoint);

    extern "C" void rte_delta_scale_2str_k(
            int* ncol, int* nlev, int* ngpt,
            Float* tau_inout, Float* ssa_inout, Float* g_inout);
}
#endif
