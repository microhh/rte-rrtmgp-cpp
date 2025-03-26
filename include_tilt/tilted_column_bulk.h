#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include "Source_functions.h"

void compress_columns_weighted_avg(const int n_x, const int n_y,  
    const int n_out, 
    const int n_tilt,
    const int compress_lay_start_idx,
    std::vector<Float>& var, std::vector<Float>& var_weighting);

void compress_columns_p_or_t(const int n_x, const int n_y, 
    const int n_out_lay,  const int n_tilt,
    const int compress_lay_start_idx,
    std::vector<Float>& var_lev, std::vector<Float>& var_lay);

void create_tilted_columns(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
        const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
        std::vector<Float>& var);

void create_tilted_columns_levlay(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
              const std::vector<Float>& zh_in, const std::vector<Float>& z_in,
              const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
              std::vector<Float>& var_lay, std::vector<Float>& var_lev);

void tilt_fields(const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1> zh, const Array<Float,1> z,
    const Array<Float,1> zh_tilt, const Array<ijk,1> path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Gas_concs& gas_concs_copy, const std::vector<std::string> gas_names
    );

void compress_fields(const int compress_lay_start_idx, const int n_col_x, const int n_col_y,
    const int n_z_in, const int n_zh_in,  const int n_z_tilt,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Gas_concs& gas_concs_copy, std::vector<std::string> gas_names);

