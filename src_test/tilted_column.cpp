#include "tilted_column.h"
#include <cmath>
#include <iostream>
#include <vector>

std::pair<Float, Float> compute_tMax_uniform(std::vector<Float>& half_grid, Float dir, Float start, int i_0){
    /*
    Find the maximum distance the ray travels before hitting a voxel boundary in that direction.
    */
    Float next_boundary;
    if (dir >= 0) {
        next_boundary = half_grid[i_0+1]; // next boundary of cell 0
    } else {
        next_boundary = half_grid[i_0];   // prev boundary
    }
    Float tMax = (dir != 0) ? std::abs((next_boundary - start) / dir)
                                : std::numeric_limits<Float>::infinity();

    return {tMax, next_boundary};
}

Float compute_tMax_not_uniform(std::vector<Float>& half_grid, Float dir_z, Float zpos, Float t, int k){
    /*
    We recompute this throughout the while loop as z spacing changes.
    */
    Float tMax;
    if (dir_z > 0) {
        tMax = t + (half_grid[k+1] - zpos) / dir_z;
    } else if (dir_z < 0) {
        tMax = t + (zpos - half_grid[k]) / std::abs(dir_z);
    } else {
        tMax = std::numeric_limits<Float>::infinity();
    }
    return tMax;
}

void tilted_path_dda(std::vector<Float>& xh, std::vector<Float>& yh,
                 std::vector<Float>& zh, std::vector<Float>& z,
                 Float sza, Float azi,
                 std::vector<ijk>& tilted_path,
                 std::vector<Float>& zh_tilted)
{
    const Float dx = xh[1]-xh[0]; 
    const Float dy = yh[1]-yh[0]; 
    const Float z_top = *std::max_element(zh.begin(), zh.end());
    const int n_x = xh.size()-1;
    const int n_y = yh.size()-1;
    const int n_z = static_cast<int>(zh.size()) - 1; // n voxels in z.
    int i = 0;
    int j = 0;
    int k = 0;

    Float dir_x = std::sin(sza) * std::sin(azi);
    Float dir_y = std::sin(sza) * std::cos(azi);
    Float dir_z = std::cos(sza);

    // starting point
    Float t = 0.0; 

    // Starting voxel
    Float x = xh[0] + dx/Float(2.);
    Float y = yh[0] + dy/Float(2.);
    Float zpos = zh[0];

    // tDeltaX determines how far along the ray we must move (in units of "t") 
    // for the horizontal component of such a movement to equal the width of a voxel
    // Amanatides & Woo “A Fast Voxel Traversal Algorithm For Ray Tracing”

    Float tDelta_x = (dir_x != 0) ? dx / std::abs(dir_x) : std::numeric_limits<Float>::infinity();
    Float tDelta_y = (dir_y != 0) ? dy / std::abs(dir_y) : std::numeric_limits<Float>::infinity();

    std::pair<Float, Float> result_x = compute_tMax_uniform(xh, dir_x, x, i);
    Float tMax_x = result_x.first;
    Float next_boundary_x = result_x.second;
    std::pair<Float, Float> result_y = compute_tMax_uniform(yh, dir_y, y, j);
    Float tMax_y = result_y.first;
    Float next_boundary_y = result_y.second;
    Float tMax_z = compute_tMax_not_uniform(zh, dir_z, zpos, t, k);

    // outputs:
    tilted_path.clear(); // this is to access cloud and gas properties later
    zh_tilted.clear();
    std::vector<Float> dz_tilted;

    while (true)
    {
        // break condition is only in z, x/y are periodic
        if (k < 0 || k >= n_z) break;

        // where is next boundary crossing?
        if (tMax_x < tMax_y && tMax_x < tMax_z) {
            // in x:
            Float dt = tMax_x - t;
            // distance to the next boundary in z:
            Float dz_segment = dt * dir_z;
            t = tMax_x;

            tilted_path.push_back({i, j, k});
            dz_tilted.push_back(dz_segment);

            if (dir_x >= 0)
                i = (i + 1) % n_x;
            else
                i = (i - 1 + n_x) % n_x;

            // Move exactly to the boundary.
            x = next_boundary_x;
            // Update next boundary and tMax_x.
            if (dir_x >= 0)
                next_boundary_x = xh[i+1];
            else
                next_boundary_x = xh[i];
            tMax_x += tDelta_x;
            y += dt * dir_y;
            zpos += dt * dir_z;
        }
        else if (tMax_y < tMax_z) {
            // in y:
            Float dt = tMax_y - t;
            Float dz_segment = dt * dir_z;
            t = tMax_y;
            tilted_path.push_back({i, j, k});
            dz_tilted.push_back(dz_segment);

            if (dir_y >= 0)
                j = (j + 1) % n_y;
            else
                j = (j - 1 + n_y) % n_y;
            y = next_boundary_y;
            if (dir_y >= 0)
                next_boundary_y = yh[j+1];
            else
                next_boundary_y = yh[j];
            tMax_y += tDelta_y;
            x += dt * dir_x;
            zpos += dt * dir_z;
        }
        else {
            // in z:
            Float dt = tMax_z - t;
            Float dz_segment = dt * dir_z;
            t = tMax_z;
            tilted_path.push_back({i, j, k});
            dz_tilted.push_back(dz_segment);

            int new_k = k + (dir_z >= 0 ? 1 : -1);
            Float current_z = zpos + dt * dir_z;
            // if we've hit the upper boundary, exit.
            if (new_k < 0 || new_k >= n_z)
                break;
            k = new_k;
            // z spacing is different now, so update tMax_z
            zpos = current_z;
            tMax_z = compute_tMax_not_uniform(zh, dir_z, zpos, t, k);
            x += dt * dir_x;
            y += dt * dir_y;
        }
    }
    zh_tilted.push_back(0.0);
    for (size_t idx = 0; idx < dz_tilted.size(); ++idx) {
        zh_tilted.push_back(zh_tilted.back() + dz_tilted[idx]);
    }
}

void tilted_path(std::vector<Float>& xh, std::vector<Float>& yh,
                 std::vector<Float>& zh, std::vector<Float>& z,
                 Float sza, Float azi,
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

    Float xp = dx/Float(2.);
    Float yp = dy/Float(2.);
    Float zp = 0.;
    Float dl = 0.;

    tilted_path.clear();
    zh_tilted.clear();
    std::vector<Float> dz_tilted;

    Float dir_x = std::sin(sza) * std::sin(azi); // azi 0 is from the north
    Float dir_y = std::sin(sza) * std::cos(azi);
    Float dir_z = std::cos(sza);  

    int z_idx = 0;
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
        Float ly = (dir_y < 0) ? (yp - yh[j]) / dir_y 
                            : (dir_y > 0) ? (yh[j+1] - yp) / dir_y 
                            : 100000.;

        // Check bounds before accessing xh[i+1]
        if (i + 1 >= xh.size()) {
            std::cerr << "Error: i+1 (" << i + 1 << ") out of bounds for xh (size=" << xh.size() << ")" << std::endl;
            break;
        }
        Float lx = (dir_x < 0) ? (xp - xh[i]) / dir_x 
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
            if(z_idx >= dz_tilted.size()){
                std::cerr << "Warning: z_idx (" << z_idx << ") out of bounds for dz_tilted (size=" << dz_tilted.size() << ")." << std::endl;
                // TODO how to handle this? push_back a new value instead of accessing an element?
                dz_tilted.push_back(dl);
                std::cout << "Pushed new dl into dz_tilted: " << dl << std::endl;
            } else {
                dz_tilted[z_idx] += dl;
            }
        }
        else
        {
            dz_tilted.push_back(dz0+dl);    
            tilted_path.push_back({i,j,k});
            z_idx += 1;
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

