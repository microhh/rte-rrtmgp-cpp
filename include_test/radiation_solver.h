/*
 * This file is imported from MicroHH (https://github.com/earth-system-radiation/earth-system-radiation)
 * and is adapted for the testing of the C++ interface to the
 * RTE+RRTMGP radiation code.
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef RADIATION_SOLVER_H
#define RADIATION_SOLVER_H

#include "array.h"
#include "gas_concs.h"
#include "gas_optics_rrtmgp.h"
#include "cloud_optics.h"
#include "aerosol_optics.h"
#include "rte_lw.h"
#include "rte_sw.h"
#include "source_functions.h"


class Radiation_solver_longwave
{
    public:
        Radiation_solver_longwave(
                const Gas_concs& gas_concs,
                const std::string& file_name_gas,
                const std::string& file_name_cloud);

        Radiation_solver_longwave(
                const Gas_concs_gpu& gas_concs,
                const std::string& file_name_gas,
                const std::string& file_name_cloud);

        void solve(
                const bool switch_fluxes,
                const bool switch_cloud_optics,
                const bool switch_output_optical,
                const bool switch_output_bnd_fluxes,
                const Gas_concs& gas_concs,
                const Array<Float,2>& p_lay, const Array<Float,2>& p_lev,
                const Array<Float,2>& t_lay, const Array<Float,2>& t_lev,
                const Array<Float,2>& col_dry,
                const Array<Float,1>& t_sfc, const Array<Float,2>& emis_sfc,
                const Array<Float,2>& lwp, const Array<Float,2>& iwp,
                const Array<Float,2>& rel, const Array<Float,2>& dei,
                Array<Float,3>& tau, Array<Float,3>& lay_source,
                Array<Float,3>& lev_source, Array<Float,2>& sfc_source,
                Array<Float,2>& lw_flux_up, Array<Float,2>& lw_flux_dn, Array<Float,2>& lw_flux_net,
                Array<Float,3>& lw_bnd_flux_up, Array<Float,3>& lw_bnd_flux_dn, Array<Float,3>& lw_bnd_flux_net) const;

        int get_n_gpt() const { return this->kdist->get_ngpt(); };
        int get_n_bnd() const { return this->kdist->get_nband(); };

        Array<int,2> get_band_lims_gpoint() const
        { return this->kdist->get_band_lims_gpoint(); }

        Array<Float,2> get_band_lims_wavenumber() const
        { return this->kdist->get_band_lims_wavenumber(); }

        #ifdef __CUDACC__
        void solve_gpu(
                const bool switch_fluxes,
                const bool switch_cloud_optics,
                const bool switch_output_optical,
                const bool switch_output_bnd_fluxes,
                const Gas_concs_gpu& gas_concs,
                const Array_gpu<Float,2>& p_lay, const Array_gpu<Float,2>& p_lev,
                const Array_gpu<Float,2>& t_lay, const Array_gpu<Float,2>& t_lev,
                const Array_gpu<Float,2>& col_dry,
                const Array_gpu<Float,1>& t_sfc, const Array_gpu<Float,2>& emis_sfc,
                const Array_gpu<Float,2>& lwp, const Array_gpu<Float,2>& iwp,
                const Array_gpu<Float,2>& rel, const Array_gpu<Float,2>& dei,
                Array_gpu<Float,3>& tau, Array_gpu<Float,3>& lay_source,
                Array_gpu<Float,3>& lev_source, Array_gpu<Float,2>& sfc_source,
                Array_gpu<Float,2>& lw_flux_up, Array_gpu<Float,2>& lw_flux_dn, Array_gpu<Float,2>& lw_flux_net,
                Array_gpu<Float,3>& lw_bnd_flux_up, Array_gpu<Float,3>& lw_bnd_flux_dn, Array_gpu<Float,3>& lw_bnd_flux_net);

        int get_n_gpt_gpu() const { return this->kdist_gpu->get_ngpt(); };
        int get_n_bnd_gpu() const { return this->kdist_gpu->get_nband(); };

        Array<int,2> get_band_lims_gpoint_gpu() const
        { return this->kdist_gpu->get_band_lims_gpoint(); }

        Array<Float,2> get_band_lims_wavenumber_gpu() const
        { return this->kdist_gpu->get_band_lims_wavenumber(); }
        #endif

    private:
        std::unique_ptr<Gas_optics_rrtmgp> kdist;
        std::unique_ptr<Cloud_optics> cloud_optics;

        #ifdef __CUDACC__
        std::unique_ptr<Gas_optics_rrtmgp_gpu> kdist_gpu;
        std::unique_ptr<Cloud_optics_gpu> cloud_optics_gpu;
        Rte_lw_gpu rte_lw;

        std::unique_ptr<Optical_props_arry_gpu> optical_props_subset;
        std::unique_ptr<Optical_props_arry_gpu> optical_props_residual;

        std::unique_ptr<Source_func_lw_gpu> sources_subset;
        std::unique_ptr<Source_func_lw_gpu> sources_residual;

        std::unique_ptr<Optical_props_1scl_gpu> cloud_optical_props_subset;
        std::unique_ptr<Optical_props_1scl_gpu> cloud_optical_props_residual;
        #endif
};


class Radiation_solver_shortwave
{
    public:
        Radiation_solver_shortwave(
                const Gas_concs& gas_concs,
                const bool switch_cloud_optics,
                const bool switch_aerosol_optics,
                const std::string& file_name_gas,
                const std::string& file_name_cloud,
                const std::string& file_name_aerosol);
        Radiation_solver_shortwave(
                const Gas_concs_gpu& gas_concs,
                const bool switch_cloud_optics,
                const bool switch_aerosol_optics,
                const std::string& file_name_gas,
                const std::string& file_name_cloud,
                const std::string& file_name_aerosol);

        void solve(
                const bool switch_fluxes,
                const bool switch_cloud_optics,
                const bool switch_aerosol_optics,
                const bool switch_output_optical,
                const bool switch_output_bnd_fluxes,
                const bool switch_delta_cloud,
                const bool switch_delta_aerosol,
                const Gas_concs& gas_concs,
                const Array<Float,2>& p_lay, const Array<Float,2>& p_lev,
                const Array<Float,2>& t_lay, const Array<Float,2>& t_lev,
                const Array<Float,2>& col_dry,
                const Array<Float,2>& sfc_alb_dir, const Array<Float,2>& sfc_alb_dif,
                const Array<Float,1>& tsi_scaling, const Array<Float,1>& mu0,
                const Array<Float,2>& lwp, const Array<Float,2>& iwp,
                const Array<Float,2>& rel, const Array<Float,2>& dei,
                const Array<Float,2>& rh,
                const Aerosol_concs& aerosol_concs,
                Array<Float,3>& tau, Array<Float,3>& ssa, Array<Float,3>& g,
                Array<Float,2>& toa_src,
                Array<Float,2>& sw_flux_up, Array<Float,2>& sw_flux_dn,
                Array<Float,2>& sw_flux_dn_dir, Array<Float,2>& sw_flux_net,
                Array<Float,3>& sw_bnd_flux_up, Array<Float,3>& sw_bnd_flux_dn,
                Array<Float,3>& sw_bnd_flux_dn_dir, Array<Float,3>& sw_bnd_flux_net) const;

        int get_n_gpt() const { return this->kdist->get_ngpt(); };
        int get_n_bnd() const { return this->kdist->get_nband(); };

        Float get_tsi() const { return this->kdist->get_tsi(); };

        Array<int,2> get_band_lims_gpoint() const
        { return this->kdist->get_band_lims_gpoint(); }

        Array<Float,2> get_band_lims_wavenumber() const
        { return this->kdist->get_band_lims_wavenumber(); }

        #ifdef __CUDACC__
        void solve_gpu(
                const bool switch_fluxes,
                const bool switch_cloud_optics,
                const bool switch_aerosol_optics,
                const bool switch_output_optical,
                const bool switch_output_bnd_fluxes,
                const bool switch_delta_cloud,
                const bool switch_delta_aerosol,
                const Gas_concs_gpu& gas_concs,
                const Array_gpu<Float,2>& p_lay, const Array_gpu<Float,2>& p_lev,
                const Array_gpu<Float,2>& t_lay, const Array_gpu<Float,2>& t_lev,
                const Array_gpu<Float,2>& col_dry,
                const Array_gpu<Float,2>& sfc_alb_dir, const Array_gpu<Float,2>& sfc_alb_dif,
                const Array_gpu<Float,1>& tsi_scaling, const Array_gpu<Float,1>& mu0,
                const Array_gpu<Float,2>& lwp, const Array_gpu<Float,2>& iwp,
                const Array_gpu<Float,2>& rel, const Array_gpu<Float,2>& dei,
                const Array_gpu<Float,2>& rh,
                const Aerosol_concs_gpu& aerosol_concs,
                Array_gpu<Float,3>& tau, Array_gpu<Float,3>& ssa, Array_gpu<Float,3>& g,
                Array_gpu<Float,2>& toa_src,
                Array_gpu<Float,2>& sw_flux_up, Array_gpu<Float,2>& sw_flux_dn,
                Array_gpu<Float,2>& sw_flux_dn_dir, Array_gpu<Float,2>& sw_flux_net,
                Array_gpu<Float,3>& sw_bnd_flux_up, Array_gpu<Float,3>& sw_bnd_flux_dn,
                Array_gpu<Float,3>& sw_bnd_flux_dn_dir, Array_gpu<Float,3>& sw_bnd_flux_net);

        int get_n_gpt_gpu() const { return this->kdist_gpu->get_ngpt(); };
        int get_n_bnd_gpu() const { return this->kdist_gpu->get_nband(); };

        Float get_tsi_gpu() const { return this->kdist_gpu->get_tsi(); };

        Array<int,2> get_band_lims_gpoint_gpu() const
        { return this->kdist_gpu->get_band_lims_gpoint(); }

        Array<Float,2> get_band_lims_wavenumber_gpu() const
        { return this->kdist_gpu->get_band_lims_wavenumber(); }
        #endif

    private:
        std::unique_ptr<Gas_optics> kdist;
        std::unique_ptr<Cloud_optics> cloud_optics;
        std::unique_ptr<Aerosol_optics> aerosol_optics;

        #ifdef __CUDACC__
        std::unique_ptr<Gas_optics_gpu> kdist_gpu;
        std::unique_ptr<Cloud_optics_gpu> cloud_optics_gpu;
        std::unique_ptr<Aerosol_optics_gpu> aerosol_optics_gpu;
        Rte_sw_gpu rte_sw;

        std::unique_ptr<Optical_props_arry_gpu> optical_props_subset;
        std::unique_ptr<Optical_props_arry_gpu> optical_props_residual;

        std::unique_ptr<Optical_props_2str_gpu> cloud_optical_props_subset;
        std::unique_ptr<Optical_props_2str_gpu> cloud_optical_props_residual;

        std::unique_ptr<Optical_props_2str_gpu> aerosol_optical_props_subset;
        std::unique_ptr<Optical_props_2str_gpu> aerosol_optical_props_residual;
        #endif
};
#endif
