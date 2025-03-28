#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include "Source_functions.h"
#include "tilted_column.h"

void tilted_path(std::vector<Float>& xh, std::vector<Float>& yh,
                 std::vector<Float>& zh, std::vector<Float>& z,
                 Float sza, Float azi,
                 Float x_start, Float y_start,
                 std::vector<ijk>& tilted_path,
                 std::vector<Float>& zh_tilted)
{
    const Float dx = xh[1]-xh[0]; 
    const Float dy = yh[1]-yh[0]; 
    const Float z_top = *std::max_element(zh.begin(), zh.end());
    const int n_x = xh.size()-1;
    const int n_y = yh.size()-1;
    int i = 0;
    int j = 0;
    int k = 0;

    Float xp = xh[0] + x_start*dx;
    Float yp = yh[0] + y_start*dy;
    Float zp = 0.;
    Float dl = 0.;

    tilted_path.clear();
    zh_tilted.clear();
    std::vector<Float> dz_tilted;

    Float dir_x = std::sin(sza) * std::sin(azi); // azi 0 is from the north
    Float dir_y = std::sin(sza) * std::cos(azi);
    Float dir_z = std::cos(sza);  

    int z_idx = -1;
    while (zp < z_top)
    {   
        // Check bounds before accessing zh[k+1]
        if (k + 1 >= zh.size()) {
            std::cerr << "Error: k+1 (" << k + 1 << ") out of bounds for zh (size=" << zh.size() << ")" << std::endl;
            break;
        }
        Float lz = (zh[k+1] - zp) / dir_z;

        // Check bounds before accessing yh[j+1]
        if (j + 1 >= yh.size()) {
            std::cerr << "Error: j+1 (" << j + 1 << ") out of bounds for yh (size=" << yh.size() << ")" << std::endl;
            break;
        }
        Float ly = (dir_y < 0) ? std::abs(yp - yh[j]) / std::abs(dir_y) 
                            : (dir_y > 0) ? (yh[j+1] - yp) / dir_y 
                            : 100000.;

        // Check bounds before accessing xh[i+1]
        if (i + 1 >= xh.size()) {
            std::cerr << "Error: i+1 (" << i + 1 << ") out of bounds for xh (size=" << xh.size() << ")" << std::endl;
            break;
        }
        Float lx = (dir_x < 0) ? std::abs(xp - xh[i]) / std::abs(dir_x) 
                            : (dir_x > 0) ? (xh[i+1] - xp) / dir_x 
                            : 100000.;


        Float l = std::min({lz, ly, lx}); 
        Float dz0 = l*dir_z;
        Float dy0 = l*dir_y;
        Float dx0 = l*dir_x;
        // Move along axes:
        zp += l*dir_z;
        yp += l*dir_y;
        xp += l*dir_x;  

        if (l < 1e-2) // ignore cell if crossed for less than 2 cm
        {
            dl = dz0/Float(2.);
            dz_tilted[z_idx] += dl;
        }
        else
        {
            dz_tilted.push_back(dz0+dl);
            tilted_path.push_back({i,j,k});
            z_idx =+ 1;
        }
        //new indices + making sure we start at hor. domain bndry once crossed
        if (l==lz)
        {
            k += 1;
        }
        else if (l==ly)
        {
            j = int(j + sign(dy0));
            j = (j == -1) ? n_y - 1 : j%n_y;
            yp = dy0 < 0 ? yh[j+1] : yh[j];
        }
        else if (l==lx)
        {
            i = int(i + sign(dx0));
            i = (i == -1) ? n_x - 1 : i%n_x;
            xp = dx0 < 0 ? xh[i+1] : xh[i];

        } 
    }
    zh_tilted.push_back(0.);
    for (int iz=0; iz<dz_tilted.size(); ++iz)
        zh_tilted.push_back(zh_tilted[iz]+dz_tilted[iz]);
}

void process_lwp_iwp(const int ix, const int iy, const int compress_lay_start_idx,
    const int n_z_in, const int n_zh_in, 
    const int n_x, const int n_y,
    const int n_z_tilt, const int n_zh_tilt,
    const Array<Float,1> zh_tilt, 
    const std::vector<ijk>& tilted_path, 
    const std::vector<Float> var,
    std::vector<Float>& var_out,
    std::vector<Float>& var_tilted)
    {

        const int n_col = n_x*n_y;
        std::vector<Float> var_tmp(n_z_tilt);
        for (int ilay=0; ilay < n_z_tilt; ++ilay)
        {
            Float dz = zh_tilt({ilay + 1}) - zh_tilt({ilay});
            const ijk offset = tilted_path[ilay];
            const int i_col_new  = ((iy+offset.j)%n_y) * n_x + ((ix + offset.i)%n_x);
            const int idx_in = offset.k + i_col_new*n_z_in;
            var_tmp[ilay] = var[idx_in] * dz;
        }

        var_tilted.resize(n_z_tilt);
        var_tilted = var_tmp;
        
        for (int ilay = 0; ilay < compress_lay_start_idx; ++ilay)
        {
            var_out[ilay] = var_tmp[ilay];
        }

        for (int ilay = compress_lay_start_idx; ilay < n_z_in; ++ilay)
        {
            const int in_offset = ilay - compress_lay_start_idx;
            const int i_lay_in = compress_lay_start_idx + 2 * in_offset;
            int num_inputs;
            if (ilay < (n_z_in - 1)) {
                num_inputs = 2;
            } 
            else {
                num_inputs = ((i_lay_in + 1) == (n_z_tilt - 1)) ? 2 : 3;
            }
            Float sum = 0.0;
            for (int k = 0; k < num_inputs; ++k)
            {
                int in_idx = i_lay_in + k;
            sum += var_tmp[in_idx];
            }
            var_out[ilay] = sum;
        }
    }

void process_w_avg_var(const int ix, const int iy, const int compress_lay_start_idx,
    const int n_z_in, const int n_zh_in, 
    const int n_x, const int n_y,
    const int n_z_tilt, const int n_zh_tilt,
    const Array<Float,1> zh_tilt, const std::vector<ijk>& tilted_path, 
    const std::vector<Float> var,
    std::vector<Float>& var_out,
    const std::vector<Float> var_weighted_tilted)
    {

        const int input_size = var.size();
        const int n_col = n_x*n_y;
        std::vector<Float> var_tmp(n_z_tilt);
        for (int ilay=0; ilay < n_z_tilt; ++ilay)
        {
            Float dz = zh_tilt({ilay + 1}) - zh_tilt({ilay});
            const ijk offset = tilted_path[ilay];
            const int i_col_new  = ((iy+offset.j)%n_y) * n_x + ((ix + offset.i)%n_x);
            const int idx_in = offset.k + i_col_new*n_z_in;
            var_tmp[ilay] = var[idx_in];
        }

        for (int ilay = 0; ilay < compress_lay_start_idx; ++ilay)
        {
            var_out[ilay] = var_tmp[ilay];

        }

        for (int ilay = compress_lay_start_idx; ilay < n_z_in; ++ilay)
        {
            const int in_offset = ilay - compress_lay_start_idx;
            const int i_lay_in = (compress_lay_start_idx + 2 * in_offset);
            int num_inputs;
            if (ilay < (n_z_in - 1)) {
                num_inputs = 2;
            } else {
                num_inputs = ((i_lay_in + 1) == (n_z_tilt - 1)) ? 2 : 3;
            }

            Float t_sum = 0.0;
            Float w_sum = 0.0;
            for (int k = 0; k < num_inputs; ++k)
            {
                int in_idx = (i_lay_in + k);
                t_sum += var_tmp[in_idx] * var_weighted_tilted[in_idx];
                w_sum += var_weighted_tilted[in_idx];
            }

            if (w_sum > 1e-6)
            {
                var_out[ilay] = t_sum / w_sum;
            } 
            else 
            {
                Float avg = 0.0;
                for (int k = 0; k < num_inputs; ++k)
                {
                    int in_idx = (i_lay_in + k);
                    avg += var_out[in_idx];
                }
                var_out[ilay] = avg / num_inputs;
            }
        }
    }

