#include "Array.h"
#include "Source_functions.h"

struct ColumnResult {
    Array<Float,2> p_lay;
    Array<Float,2> t_lay;
    Array<Float,2> p_lev;
    Array<Float,2> t_lev;
    Array<Float,2> lwp;
    Array<Float,2> iwp;
    Array<Float,2> rel;
    Array<Float,2> dei;
    Array<Float,2> h2o;
    Array<Float,2> o3;
};

void tilt_fields_single_column(const int idx_col_x, const int idx_col_y,
    const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1> zh, const Array<Float,1> z,
    const Array<Float,1> zh_tilt, const Array<ijk,1> path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Array<Float,2>* h2o_copy, Array<Float,2>* o3_copy, 
    const bool switch_cloud_optics, const bool switch_liq_cloud_optics, const bool switch_ice_cloud_optics
);

void compress_fields_single_column(const int compress_lay_start_idx, const int n_col_x, const int n_col_y,
    const int n_z_in, const int n_zh_in,  const int n_z_tilt,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Array<Float,2>* h2o_copy, Array<Float,2>* o3_copy, 
    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics);

void post_process_output(const std::vector<ColumnResult>& col_results,
                         const int n_col_x, const int n_col_y,
                         const int n_z, const int n_zh,
                         Array<Float,2>* p_lay_out,
                         Array<Float,2>* t_lay_out,
                         Array<Float,2>* p_lev_out,
                         Array<Float,2>* t_lev_out,
                         Array<Float,2>* lwp_out,
                         Array<Float,2>* iwp_out,
                         Array<Float,2>* rel_out,
                         Array<Float,2>* dei_out,
                         Gas_concs& gas_concs_out,
                         const std::vector<std::string>& gas_names,
                         const bool switch_cloud_optics,
                         const bool switch_liq_cloud_optics,
                         const bool switch_ice_cloud_optics);
