#ifndef TILTED_COLUMN_H
#define TILTED_COLUMN_H

#include "Array.h"

struct ijk
{
    int i;
    int j;
    int k;
};

template<typename TF> int sign(TF value)
{
    return (TF(0.) < value) - (value < TF(0.));
}

template<typename TF>
void tilted_path(std::vector<TF>& xh, std::vector<TF>& yh,
                 std::vector<TF>& zh, std::vector<TF>& z,
                 TF sza, TF azi,
                 std::vector<ijk>& tilted_path,
                 std::vector<TF>& dz_tilted);

template<typename TF>
void create_tilted_columns(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                           const std::vector<TF>& zh_tilted, const std::vector<ijk>& tilted_path,
                           std::vector<TF>& var);

template<typename TF>
void create_tilted_columns_levlay(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                                  const std::vector<TF>& zh_in, const std::vector<TF>& z_in,
                                  const std::vector<TF>& zh_tilted, const std::vector<ijk>& tilted_path,
                                  std::vector<TF>& var_lay, std::vector<TF>& var_lev);

template<typename TF>
void interpolate(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                 const std::vector<TF>& zh_in, const std::vector<TF>& z_in,
                 const std::vector<TF>& play_in, const std::vector<TF>& plev_in,
                 const TF zp, const ijk offset,
                 TF* p_out);

template<typename TF>
void translate_heating_rates(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                             const std::vector<ijk>& tilted_path, std::vector<TF>& heat);









#endif
