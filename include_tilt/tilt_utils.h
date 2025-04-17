#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "Array.h"
#include "Source_functions.h"

struct ColumnResult {
    Array<Float,1> lwp;
    Array<Float,1> iwp;
    Array<Float,1> rel;
    Array<Float,1> dei;
};

struct ijk
{
    int i;
    int j;
    int k;
};

inline int sign(Float value)
{
    return (Float(0.) < value) - (value < Float(0.));
}

void tilted_path(std::vector<Float>& xh, std::vector<Float>& yh,
        std::vector<Float>& zh, std::vector<Float>& z,
        Float sza, Float azi,
        Float x_start, Float y_start,
        std::vector<ijk>& tilted_path,
        std::vector<Float>& zh_tilted);

void post_process_output(const std::vector<ColumnResult>& col_results,
        const int n_col_x, const int n_col_y,
        const int n_z, const int n_zh,
        Array<Float,2>* lwp_out,
        Array<Float,2>* iwp_out,
        Array<Float,2>* rel_out,
        Array<Float,2>* dei_out,
        const bool switch_liq_cloud_optics,
        const bool switch_ice_cloud_optics);

void restore_bkg_profile(const int n_x, const int n_y, 
                      const int n_full,
                      const int n_tilt, 
                      const int bkg_start, 
                      std::vector<Float>& var,
                      std::vector<Float>& var_w_bkg);

void restore_bkg_profile_bundle(const int n_col_x, const int n_col_y, 
    const int n_lay, const int n_lev, 
    const int n_lay_tot, const int n_lev_tot, 
    const int n_z_in, const int n_zh_in,
    const int bkg_start_z, const int bkg_start_zh, 
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, Array<Float,2>* rh_copy, 
    Gas_concs& gas_concs_copy, Aerosol_concs& aerosol_concs_copy,
    Array<Float,2>* p_lay, Array<Float,2>* t_lay, Array<Float,2>* p_lev, Array<Float,2>* t_lev, 
    Array<Float,2>* lwp, Array<Float,2>* iwp, Array<Float,2>* rel, Array<Float,2>* dei, Array<Float,2>* rh, 
    Gas_concs& gas_concs, Aerosol_concs& aerosol_concs, 
    std::vector<std::string> gas_names, std::vector<std::string> aerosol_names,
    bool switch_liq_cloud_optics, bool switch_ice_cloud_optics, bool switch_aerosol_optics
    );

void compress_columns_weighted_avg(const int n_x, const int n_y,  
                      const int n_out, 
                      const int n_tilt,
                      const int compress_lay_start_idx,
                      std::vector<Float>& var, std::vector<Float>& var_weighting);

void compress_columns_p_or_t(const int n_x, const int n_y, 
                      const int n_out_lay,  const int n_tilt,
                      const int compress_lay_start_idx,
                      std::vector<Float>& var_lev, std::vector<Float>& var_lay);

void tilt_fields(const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1> zh, const Array<Float,1> z,
    const Array<Float,1> zh_tilt, const Array<ijk,1> path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* rh_copy, 
    Gas_concs& gas_concs_copy, const std::vector<std::string> gas_names,
    Aerosol_concs& aerosol_concs_copy, const std::vector<std::string> aerosol_names, const bool switch_aerosol_optics
    );

void compress_fields(const int compress_lay_start_idx, const int n_col_x, const int n_col_y,
    const int n_z_in, const int n_zh_in,  const int n_z_tilt,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* rh_copy, 
    Gas_concs& gas_concs_copy, std::vector<std::string> gas_names,
    Aerosol_concs& aerosol_concs_copy, std::vector<std::string> aerosol_names, const bool switch_aerosol_optics);

void create_tilted_columns(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                           const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                           std::vector<Float>& var);

void interpolate(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                 const std::vector<Float>& zh_in, const std::vector<Float>& zf_in,
                 const std::vector<Float>& play_in, const std::vector<Float>& plev_in,
                 const Float zp, const ijk offset,
                 Float* p_out);

void create_tilted_columns_levlay(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                                 const std::vector<Float>& zh_in, const std::vector<Float>& z_in,
                                 const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                                 std::vector<Float>& var_lay, std::vector<Float>& var_lev);

void tica_tilt(const Float sza, const Float azi,
    const int n_col_x, const int n_col_y, const int n_col,
    const int n_lay, const int n_lev, const int n_z_in, const int n_zh_in ,
    Array<Float,1> xh, Array<Float,1> yh, Array<Float,1> zh, Array<Float,1> z,
    Array<Float,2> p_lay, Array<Float,2> t_lay, Array<Float,2> p_lev, Array<Float,2> t_lev, 
    Array<Float,2> lwp, Array<Float,2> iwp, Array<Float,2> rel, Array<Float,2> dei, Array<Float,2> rh, 
    Gas_concs gas_concs, Aerosol_concs aerosol_concs,
    Array<Float,2>& p_lay_out, Array<Float,2>& t_lay_out, Array<Float,2>& p_lev_out, Array<Float,2>& t_lev_out, 
    Array<Float,2>& lwp_out, Array<Float,2>& iwp_out, Array<Float,2>& rel_out, Array<Float,2>& dei_out, Array<Float,2>& rh_out, 
    Gas_concs& gas_concs_out, Aerosol_concs aerosol_concs_out,
    std::vector<std::string> gas_names, std::vector<std::string> aerosol_names,
    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics, bool switch_aerosol_optics);
