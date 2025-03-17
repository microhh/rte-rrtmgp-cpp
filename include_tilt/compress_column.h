#include "Array.h"
#include "Source_functions.h"

void compress_columns(const int n_x, const int n_y, 
                      const int n_out, const int n_tilt,
                      const int compress_lay_start_idx,
                      std::vector<Float>& var);

void compress_columns_weighted_avg(const int n_x, const int n_y,  
                      const int n_out, const int n_tilt,
                      const int compress_lay_start_idx,
                      std::vector<Float>& var, std::vector<Float>& var_weighting);

void compress_columns_p_or_t(const int n_x, const int n_y, 
                      const int n_out_lay, const int n_tilt,
                      const int compress_lay_start_idx,
                      std::vector<Float>& var_lev, std::vector<Float>& var_lay);
