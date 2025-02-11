#include "tilted_column.h"
#include <cmath>
#include <iostream>
#include <vector>

template<typename TF>
void tilted_path(std::vector<TF>& xh, std::vector<TF>& yh,
                 std::vector<TF>& zh, std::vector<TF>& z,
                 TF sza, TF azi,
                 std::vector<ijk>& tilted_path,
                 std::vector<TF>& zh_tilted)
{
    const TF dx = xh[1]-xh[0]; 
    const TF dy = yh[1]-yh[0]; 
    const TF z_top = *std::max_element(zh.begin(), zh.end());
    const TF azi_deg = azi*TF(360.)/TF(2*3.14159);
    const int n_x = xh.size()-1;
    const int n_y = yh.size()-1;
    int i = 0;
    int j = 0;
    int k = 0;

    TF xp = dx/TF(2.);
    TF yp = dy/TF(2.);
    TF zp = 0.;
    TF dl = 0.;

    tilted_path.clear();
    zh_tilted.clear();
    std::vector<TF> dz_tilted;

    TF dir_x = std::sin(sza) * std::cos(azi);
    TF dir_y = std::sin(sza) * std::sin(azi);
    TF dir_z = std::cos(sza);  

    int z_idx = 0;
    while (zp < z_top)
    {   
        // Check bounds before accessing zh[k+1]
        if (k + 1 >= zh.size()) {
            std::cerr << "Error: k+1 (" << k + 1 << ") out of bounds for zh (size=" << zh.size() << ")" << std::endl;
            break;
        }
        TF lz = (zh[k+1] - zp) / dir_z;

        // Check bounds before accessing yh[j+1]
        if (j + 1 >= yh.size()) {
            std::cerr << "Error: j+1 (" << j + 1 << ") out of bounds for yh (size=" << yh.size() << ")" << std::endl;
            break;
        }
        TF ly = (dir_y < 0) ? (yp - yh[j]) / dir_y 
                            : (dir_y > 0) ? (yh[j+1] - yp) / dir_y 
                            : 100000.;

        // Check bounds before accessing xh[i+1]
        if (i + 1 >= xh.size()) {
            std::cerr << "Error: i+1 (" << i + 1 << ") out of bounds for xh (size=" << xh.size() << ")" << std::endl;
            break;
        }
        TF lx = (dir_x < 0) ? (xp - xh[i]) / dir_x 
                            : (dir_x > 0) ? (xh[i+1] - xp) / dir_x 
                            : 100000.;


        TF l = std::min({lz, ly, lx}); 
        TF dz0 = l*std::cos(sza);
        TF dy0 = l*std::sin(sza)*std::cos(azi);
        TF dx0 = l*std::sin(sza)*std::sin(azi);
        // Move along axes:
        zp += dz0;
        yp += dy0;
        xp += dx0;   

        if (l < 1e-2) // ignore cell if crossed for less than 2 cm
        {
            
            dl = dz0/TF(2.);
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


template<typename TF>
void create_tilted_columns(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                           const std::vector<TF>& zh_tilted, const std::vector<ijk>& tilted_path,
                           std::vector<TF>& var)
{
    const int n_lay = tilted_path.size();
    const int n_lev = zh_tilted.size();

    std::vector<TF> var_tmp(n_lay*n_x*n_y);

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

template<typename TF>
void interpolate(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                 const std::vector<TF>& zh_in, const std::vector<TF>& zf_in,
                 const std::vector<TF>& play_in, const std::vector<TF>& plev_in,
                 const TF zp, const ijk offset,
                 TF* p_out)
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
        const TF* p_top;
        const TF* p_bot;
        TF  z_top;
        TF  z_bot;

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

        TF dz = z_top-z_bot;

        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x;
                const TF pres = (zp-z_bot)/dz*p_top[idx_in] + (z_top-zp)/dz*p_bot[idx_in];
                p_out[idx_out] = pres;
            } 

    }
}

template<typename TF>
void create_tilted_columns_levlay(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                                 const std::vector<TF>& zh_in, const std::vector<TF>& z_in,
                                 const std::vector<TF>& zh_tilted, const std::vector<ijk>& tilted_path,
                                 std::vector<TF>& var_lay, std::vector<TF>& var_lev)

{
    const int n_lay = tilted_path.size();
    const int n_lev = zh_tilted.size();
    std::vector<TF> z_tilted(n_lay);
    for (int ilay=0; ilay<n_lay; ++ilay)
        z_tilted[ilay] = (zh_tilted[ilay]+zh_tilted[ilay+1])/TF(2.);

    std::vector<TF> var_lay_tmp(n_lay*n_x*n_y);
    std::vector<TF> var_lev_tmp(n_lev*n_x*n_y);
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

template<typename TF>
void translate_heating_rates(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                             const std::vector<ijk>& tilted_path, std::vector<TF>& heat)
{
    const int n_lay = tilted_path.size();

    std::vector<TF> heat_tmp(n_lay_in*n_x*n_y, TF(0.));

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
                    heat_tmp[idx_out] /= TF(n);
                }
        }
    }
    heat.resize(heat_tmp.size());
    heat = heat_tmp;
}   

#ifdef FLOAT_SINGLE_RRTMGP
template void tilted_path<float>(std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, float, float, std::vector<ijk>&, std::vector<float>&);

template void create_tilted_columns(const int, const int, const int, const int,
                           const std::vector<float>&, const std::vector<ijk>&,
                           std::vector<float>&);

template void create_tilted_columns_levlay(const int, const int, const int, const int,
                                           const std::vector<double>&,const std::vector<double>&,
                                           const std::vector<double>&, const std::vector<ijk>&,
                                           std::vector<double>&, std::vector<double>&);
template void translate_heating_rates(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                             const std::vector<ijk>& tilted_path, std::vector<float>& heat);
#else
template void tilted_path<double>(std::vector<double>&, std::vector<double>&, std::vector<double>&, std::vector<double>&, double, double, std::vector<ijk>&, std::vector<double>&);

template void create_tilted_columns(const int, const int, const int, const int,
                           const std::vector<double>&, const std::vector<ijk>&,
                           std::vector<double>&);

template void create_tilted_columns_levlay(const int, const int, const int, const int,
                                           const std::vector<double>&,const std::vector<double>&,
                                           const std::vector<double>&, const std::vector<ijk>&,
                                           std::vector<double>&, std::vector<double>&);

template void translate_heating_rates(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                             const std::vector<ijk>& tilted_path, std::vector<double>& heat);
//template void interpolate_pressures(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
//                                    const std::vector<double>& zh_in, const std::vector<double>& z_in,
//                                    const std::vector<double>& play_in, const std::vector<double>& plev_in,
//                                    const double zp, const ijk offset,
//                                    double* p_out);
#endif


