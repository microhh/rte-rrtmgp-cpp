#include "tilted_column.h"
#include <cmath>
#include <iostream>
#include <vector>

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

void create_tilted_columns(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                           const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                           std::vector<Float>& var)
{
    const int n_lay = tilted_path.size();
    const int n_lev = zh_tilted.size();

    std::vector<Float> var_tmp(n_lay*n_x*n_y);

    #pragma omp parallel for
    for (int ilay=0; ilay<n_lay; ++ilay)
    {
        const ijk offset = tilted_path[ilay];
        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y + ilay*n_y*n_x;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + offset.k*n_y*n_x;
                var_tmp[idx_out] = var[idx_in];
            } 
    }

    var.resize(var_tmp.size());
    var = var_tmp;
}

void interpolate_col(const int n_in, const int n_out,
                    const std::vector<Float>& z_in,
                    const std::vector<Float>& z_out, 
                    const std::vector<Float>& var_lay_in, 
                    Float* var_out)
{
    int idx2; int idx1; Float z1; Float z2; Float v1; Float v2; Float z;
    Float val;
    for (int i=0; i < n_out; ++i)
    {
        z = z_out[i];
        auto it = std::upper_bound(z_in.begin(), z_in.end(), z);
        idx2 = std::distance(z_in.begin(), it);
        if (idx2 == 0) {
            idx1 = 0;
            idx2 = 1;
        } else if (idx2 >= n_in) {
            idx1 = n_in - 2;
            idx2 = n_in - 1;
        } else {
            idx1 = idx2 - 1;
        }

        z1 = z_in[idx1];
        z2 = z_in[idx2];
        v1 = var_lay_in[idx1];
        v2 = var_lay_in[idx2];

        // Propagate nans
        if (!std::isfinite(z1) || !std::isfinite(z2) || 
            !std::isfinite(v1) || !std::isfinite(v2)) {
            var_out[i] = std::numeric_limits<Float>::quiet_NaN();
            continue;
        }
        if (z2 == z1) {
            var_out[i] = v1;
            continue;
        }

        val = (v1*(z2 - z) + v2*(z - z1))/(z2 - z1);

        if (!std::isfinite(val) || std::abs(val) > 1e8 * std::max(std::abs(v1), std::abs(v2))) {
            val = std::numeric_limits<Float>::quiet_NaN();
        }
        var_out[i] = val;
    }
}

void weighted_avg_col(const int n_in, const int n_out,
                      const std::vector<Float>& z_in,
                      const std::vector<Float>& z_out, 
                      const std::vector<Float>& var_lay_in, 
                      Float* var_out)
{
    for (int i = 0; i < n_out; ++i) {

        if (i == n_out - 1) {
            var_out[i] = var_lay_in.back();
            continue;
        }

        Float z_start = z_out[i];
        Float z_end = (i < n_out - 1) ? z_out[i + 1] : z_in.back();

        int idx_start = std::upper_bound(z_in.begin(), z_in.end() - 1, z_start) - z_in.begin() - 1;
        if (idx_start < 0) { 
            idx_start = 0;
        }
        int idx_end = std::lower_bound(z_in.begin() + 1, z_in.end(), z_end) - z_in.begin();
        if (idx_end < 1) { 
            idx_end = 1;
        }
        if (idx_start == idx_end - 1) {
            var_out[i] = var_lay_in[idx_start];
            continue;
        }

        Float weighted_avg = 0.0;
        Float dz_total = 0.0;

        for (int j = idx_start; j < idx_end; ++j) {
            Float cell_lower = z_in[j];
            Float cell_upper = z_in[j + 1];
            Float lower = std::max(z_start, cell_lower);
            Float upper = std::min(z_end, cell_upper);
            Float overlap = upper - lower;
            if (overlap > 0) {
                weighted_avg += var_lay_in[j] * overlap;
                dz_total += overlap;
            }
        }

        if (dz_total > 0.0) {
            weighted_avg /= dz_total;
        } else {
            weighted_avg = std::nan("");
        }

        var_out[i] = weighted_avg;
    }
}

void interpolate_3D_field(const int n_x, const int n_y,
                            const std::vector<Float>& z_in,
                            const std::vector<Float>& z_out,
                            std::vector<Float>& var_in
                            )
{
    const int n_col = n_x*n_y;
    const int n_in = z_in.size();
    const int n_out = z_out.size();
    std::vector<Float> var_tmp(n_out*n_col);

    #pragma omp parallel for
    for (int i = 0; i < n_col; i++) 
    {
        std::vector<Float> col_in(n_in);
        for (int j = 0; j < n_in; j++)
        {
            col_in[j] = var_in[i*n_in + j];
        }
        weighted_avg_col(n_in, n_out, z_in, z_out, col_in, &var_tmp[i*n_out]);
    }

    var_in.resize(n_out*n_col);
    var_in = var_tmp;
}

void interpolate(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                 const std::vector<Float>& zh_in, const std::vector<Float>& zf_in,
                 const std::vector<Float>& play_in, const std::vector<Float>& plev_in,
                 const Float zp, const ijk offset,
                 Float* p_out)
{
    int zp_in_zh = -1; // half level
    int zp_in_zf = -1; // full level
    for (int ilev=0; ilev<n_lev_in; ++ilev)
        if (std::abs(zh_in[ilev]-zp) < 1e-2)
            zp_in_zh= ilev;
    for (int ilay=0; ilay<n_lay_in; ++ilay)
        if (std::abs(zf_in[ilay]-zp) < 1e-2)
            zp_in_zf = ilay;
    if (zp_in_zh > -1)
    {
        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + zp_in_zh*n_y*n_x;
                p_out[idx_out] = plev_in[idx_in];
            } 
    }
    else if (zp_in_zf > -1)
    {
        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + zp_in_zf*n_y*n_x;
                p_out[idx_out] = play_in[idx_in];
            } 
    }
    else
    {
        int posh_bot = 0;
        int posf_bot = 0;
        for (int ilev=0; ilev<n_lev_in-1; ++ilev)
            if (zh_in[ilev] < zp)
                posh_bot = ilev;     
        for (int ilay=0; ilay<n_lay_in; ++ilay)
            if (zf_in[ilay] < zp)
                posf_bot = ilay;         
        const Float* p_top;
        const Float* p_bot;
        Float  z_top;
        Float  z_bot;

        const int zh_top = zh_in[posh_bot+1];
        const int zh_bot = zh_in[posh_bot];
        const int zf_top = (posf_bot+1 < n_lay_in) ? zf_in[posf_bot+1] : zh_top+1;
        const int zf_bot = zf_in[posf_bot];

        if (zh_top > zf_top)
        {
            p_top = &play_in.data()[(posf_bot+1)*n_x*n_y];       
            z_top = zf_in[posf_bot+1];       
        }
        else
        {   
            p_top = &plev_in.data()[(posh_bot+1)*n_x*n_y];       
            z_top = zh_in[posh_bot+1];
        }
        if (zh_bot < zf_bot)
        {
            p_bot = &play_in.data()[(posf_bot)*n_x*n_y];       
            z_bot = zf_in[posf_bot];       
        }
        else
        {   
            p_bot = &plev_in.data()[(posh_bot)*n_x*n_y];       
            z_bot = zh_in[posh_bot];
        }

        Float dz = z_top-z_bot;

        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x;
                const Float pres = (zp-z_bot)/dz*p_top[idx_in] + (z_top-zp)/dz*p_bot[idx_in];
                p_out[idx_out] = pres;
            } 

    }
}

void create_tilted_columns_levlay(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                                 const std::vector<Float>& zh_in, const std::vector<Float>& z_in,
                                 const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                                 std::vector<Float>& var_lay, std::vector<Float>& var_lev)

{
    const int n_lay = tilted_path.size();
    const int n_lev = zh_tilted.size();
    std::vector<Float> z_tilted(n_lay);
    for (int ilay=0; ilay<n_lay; ++ilay)
        z_tilted[ilay] = (zh_tilted[ilay]+zh_tilted[ilay+1])/Float(2.);

    std::vector<Float> var_lay_tmp(n_lay*n_x*n_y);
    std::vector<Float> var_lev_tmp(n_lev*n_x*n_y);
    for (int iy=0; iy<n_y; ++iy)
        for (int ix=0; ix<n_x; ++ix)
        {
            const int idx = ix + iy*n_y;
            var_lev_tmp[idx] = var_lev[idx];
        } 

    #pragma omp parallel for
    for (int ilev=1; ilev<n_lev; ++ilev)
        interpolate(n_x, n_y, n_lay_in, n_lev_in, zh_in, z_in, var_lay, var_lev, zh_tilted[ilev], tilted_path[ilev-1], &var_lev_tmp.data()[ilev*n_y*n_x]);
    #pragma omp parallel for
    for (int ilay=0; ilay<n_lay; ++ilay)
        interpolate(n_x, n_y, n_lay_in, n_lev_in, zh_in, z_in, var_lay, var_lev, z_tilted[ilay], tilted_path[ilay], &var_lay_tmp.data()[ilay*n_y*n_x]);

    var_lay.resize(n_lay*n_y*n_x);
    var_lev.resize(n_lev*n_y*n_x);
    var_lay = var_lay_tmp;
    var_lev = var_lev_tmp;
}

void translate_heating_rates(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                             const std::vector<ijk>& tilted_path, std::vector<Float>& heat)
{
    const int n_lay = tilted_path.size();

    std::vector<Float> heat_tmp(n_lay_in*n_x*n_y, Float(0.));

    int k = 0;
    for (int ilay=0; ilay<n_lay_in; ++ilay)
    {
        int n = 0;
        while (tilted_path[k].k == ilay)
        {
            const ijk offset = tilted_path[k];

            for (int iy=0; iy<n_y; ++iy)
                for (int ix=0; ix<n_x; ++ix)
                {
                    const int idx_out  = ix + iy*n_y + ilay*n_y*n_x;
                    const int idx_in = (ix - offset.i)%n_x + (iy-offset.j)%n_y * n_x + k*n_y*n_x;
                    heat_tmp[idx_out] += heat[idx_in];
                }
            n += 1;
            k += 1;
        }
        if (n>1)
        {
            for (int iy=0; iy<n_y; ++iy)
                for (int ix=0; ix<n_x; ++ix)
                {
                    const int idx_out  = ix + iy*n_y + ilay*n_y*n_x;
                    heat_tmp[idx_out] /= Float(n);
                }
        }
    }
    heat.resize(heat_tmp.size());
    heat = heat_tmp;
}   

