#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include "Source_functions.h"

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
                 std::vector<Float>& dz_tilted);

void process_liq_or_ice(const int ix, const int iy, const int compress_lay_start_idx,
        const int n_z_in, const int n_zh_in, 
        const int n_x, const int n_y,
        const int n_z_tilt, const int n_zh_tilt,
        const Array<Float,1> zh_tilt, 
        const std::vector<ijk>& tilted_path, 
        const std::vector<Float> var_wp,
        std::vector<Float>& var_wp_out,
        const std::vector<Float> var_size,
        std::vector<Float>& var_size_out);
