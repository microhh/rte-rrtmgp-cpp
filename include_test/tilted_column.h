#ifndef RADIATION_SOLVER_H
#define RADIATION_SOLVER_H

#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics_rrtmgp.h"
#include "Cloud_optics.h"
#include "Aerosol_optics.h"
#include "Rte_lw.h"
#include "Rte_sw.h"
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

void create_tilted_columns(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                           const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                           std::vector<Float>& var);

void create_tilted_columns_levlay(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                                  const std::vector<Float>& zh_in, const std::vector<Float>& z_in,
                                  const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                                  std::vector<Float>& var_lay, std::vector<Float>& var_lev);

void interpolate(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                 const std::vector<Float>& zh_in, const std::vector<Float>& z_in,
                 const std::vector<Float>& play_in, const std::vector<Float>& plev_in,
                 const Float zp, const ijk offset,
                 Float* p_out);

void interpolate_col(const int n_in, const int n_out,
                    const std::vector<Float>& z_in,
                    const std::vector<Float>& z_out, 
                    const std::vector<Float>& var_lay_in, 
                    Float* var_out);


void weighted_avg_col(const int n_in, const int n_out,
                    const std::vector<Float>& z_in,
                    const Float z_out, 
                    const std::vector<Float>& var_lay_in, 
                    Float* var_out);

void interpolate_3D_field(const int n_x, const int n_y,
                            const std::vector<Float>& z_in,
                            const std::vector<Float>& z_out,
                            std::vector<Float>& var_in);

void translate_heating_rates(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                             const std::vector<ijk>& tilted_path, std::vector<Float>& heat);

#endif
