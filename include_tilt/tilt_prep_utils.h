#include "Array.h"
#include "Source_functions.h"

void read_and_set_vmr(
        const std::string& gas_name, const int n_col_x, const int n_col_y, const int n_lay,
        const Netcdf_handle& input_nc, Gas_concs& gas_concs);

bool parse_command_line_options(
        std::map<std::string, std::pair<bool, std::string>>& command_line_switches,
        std::map<std::string, std::pair<int, std::string>>& command_line_ints,
        int argc, char** argv);

void print_command_line_options(
        const std::map<std::string, std::pair<bool, std::string>>& command_line_switches,
        const std::map<std::string, std::pair<int, std::string>>& command_line_ints);

std::vector<Float> linspace(Float start, Float end, int num_points);

bool prepare_netcdf(Netcdf_handle& input_nc, std::string file_name, int n_lay, int n_lev, int n_col_x, int n_col_y,
                    int n_zh, int n_z, 
                    Float sza, std::vector<Float> zh, std::vector<Float> z,
                    Array<Float,2> p_lay, Array<Float,2> t_lay, Array<Float,2> p_lev, Array<Float,2> t_lev, 
                    Array<Float,2> lwp, Array<Float,2> iwp, Array<Float,2> rel, Array<Float,2> dei, 
                    Gas_concs& gas_concs, std::vector<std::string> gas_names,
                    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics);


void tilt_fields(const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1> zh, const Array<Float,1> z,
    const Array<Float,1> zh_tilt, const Array<ijk,1> path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Gas_concs& gas_concs_copy, const std::vector<std::string> gas_names,
    const bool switch_cloud_optics, const bool switch_liq_cloud_optics, const bool switch_ice_cloud_optics
);

void compress_fields(const int compress_lay_start_idx, const int n_col_x, const int n_col_y,
    const int n_z_in, const int n_zh_in,  const int n_z_tilt,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Gas_concs& gas_concs_copy, std::vector<std::string> gas_names,
    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics);

void restore_bkg_profile(const int n_x, const int n_y, 
                      const int n_full,
                      const int n_tilt, 
                      const int bkg_start, 
                      std::vector<Float>& var,
                      std::vector<Float>& var_w_bkg);

void restore_bkg_profile_bundle(const int n_col_x, const int n_col_y, const int n_lay, const int n_lev, 
    const int n_lay_tot, const int n_lev_tot, 
    const int n_z_in, const int n_zh_in, const int bkg_start_z, const int bkg_start_zh, 
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Gas_concs& gas_concs_copy,
    Array<Float,2>* p_lay, Array<Float,2>* t_lay, Array<Float,2>* p_lev, Array<Float,2>* t_lev, 
    Array<Float,2>* lwp, Array<Float,2>* iwp, Array<Float,2>* rel, Array<Float,2>* dei, 
    Gas_concs& gas_concs, 
    std::vector<std::string> gas_names,
    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics
);

