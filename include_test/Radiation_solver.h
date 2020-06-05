/*
 * This file is imported from MicroHH (https://github.com/microhh/microhh)
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

#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics_rrtmgp.h"

template<typename TF>
class Radiation_solver_longwave
{
    public:
        Radiation_solver_longwave(const Gas_concs<TF>& gas_concs, const std::string& file_name);

        void solve(
                const bool sw_output_optical,
                const bool sw_output_bnd_fluxes,
                const Gas_concs<TF>& gas_concs,
                const Array<TF,2>& p_lay, const Array<TF,2>& p_lev,
                const Array<TF,2>& t_lay, const Array<TF,2>& t_lev,
                const Array<TF,2>& col_dry,
                const Array<TF,1>& t_sfc, const Array<TF,2>& emis_sfc,
                Array<TF,3>& tau, Array<TF,3>& lay_source,
                Array<TF,3>& lev_source_inc, Array<TF,3>& lev_source_dec, Array<TF,2>& sfc_source,
                Array<TF,2>& lw_flux_up, Array<TF,2>& lw_flux_dn, Array<TF,2>& lw_flux_net,
                Array<TF,3>& lw_bnd_flux_up, Array<TF,3>& lw_bnd_flux_dn, Array<TF,3>& lw_bnd_flux_net) const;

        int get_n_gpt() const { return this->kdist->get_ngpt(); };
        int get_n_bnd() const { return this->kdist->get_nband(); };

        Array<int,2> get_band_lims_gpoint() const
        { return this->kdist->get_band_lims_gpoint(); }

        Array<TF,2> get_band_lims_wavenumber() const
        { return this->kdist->get_band_lims_wavenumber(); }

    private:
        std::unique_ptr<Gas_optics_rrtmgp<TF>> kdist;
};

template<typename TF>
class Radiation_solver_shortwave
{
    public:
        Radiation_solver_shortwave(const Gas_concs<TF>& gas_concs, const std::string& file_name);

        void solve(
                const bool sw_output_optical,
                const bool sw_output_bnd_fluxes,
                const Gas_concs<TF>& gas_concs,
                const Array<TF,2>& p_lay, const Array<TF,2>& p_lev,
                const Array<TF,2>& t_lay, const Array<TF,2>& t_lev,
                const Array<TF,2>& col_dry,
                const Array<TF,2>& sfc_alb_dir, const Array<TF,2>& sfc_alb_dif,
                const Array<TF,1>& tsi_scaling, const Array<TF,1>& mu0,
                Array<TF,3>& tau, Array<TF,3>& ssa, Array<TF,3>& g,
                Array<TF,2>& toa_src,
                Array<TF,2>& sw_flux_up, Array<TF,2>& sw_flux_dn,
                Array<TF,2>& sw_flux_dn_dir, Array<TF,2>& sw_flux_net,
                Array<TF,3>& sw_bnd_flux_up, Array<TF,3>& sw_bnd_flux_dn,
                Array<TF,3>& sw_bnd_flux_dn_dir, Array<TF,3>& sw_bnd_flux_net) const;

        int get_n_gpt() const { return this->kdist->get_ngpt(); };
        int get_n_bnd() const { return this->kdist->get_nband(); };

        TF get_tsi() const { return this->kdist->get_tsi(); };

        Array<int,2> get_band_lims_gpoint() const
        { return this->kdist->get_band_lims_gpoint(); }

        Array<TF,2> get_band_lims_wavenumber() const
        { return this->kdist->get_band_lims_wavenumber(); }

    private:
        std::unique_ptr<Gas_optics<TF>> kdist;
};
#endif
