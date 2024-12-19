import numpy as np
import netCDF4 as nc

float_type = "f8"
n_col_y = 1
n_col_x = 24
n_lay = 72
n_bnd_lw = 16
n_bnd_sw = 14

nc_file = nc.Dataset("rte_rrtmgp_input.nc", mode="w", datamodel="NETCDF4", clobber=True)

# Radiation profiles.
z_top = 70.e3
z_trop = 15.e3

zh = np.zeros(n_lay+1)
for i in range(1, n_lay//2+1):
    zh[i           ] =          2.*i*(z_trop        )/n_lay
    zh[i + n_lay//2] = z_trop + 2.*i*(z_top - z_trop)/n_lay

z = 0.5*(zh[1:] + zh[:-1])

def calc_p_q_T(z):
    q_0 = 0.01864 # for 300 K SST.
    z_q1 = 4.0e3
    z_q2 = 7.5e3
    q_t = 1.e-8

    q = q_0 * np.exp(-z/z_q1) * np.exp(-(z/z_q2)**2)

    i_above_zt = np.where(z > z_trop)
    q[i_above_zt] = q_t

    T_0 = 300.
    Tv_0 = (1. + 0.608*q_0)*T_0

    # This is the lapse rate of Tv
    gamma = 6.7e-3
    # I follow here Robert's approach, which gives rather different pressures.

    # Tv = Tv_0 - gamma*z
    T = T_0 - gamma*z / (1. + 0.608*q)

    # Tv_t = Tv_0 - gamma*z_trop
    # Tv[i_above_zt] = Tv_t

    # CvH: The q_0 here is wrong
    T[i_above_zt] = T_0 - gamma*z_trop / (1. + 0.608*q_0)

    Tv = T * (1. + 0.608*q)

    # T = Tv / (1. + 0.608*q)

    g = 9.79764
    Rd = 287.04
    p0 = 101480.

    p = p0 * (Tv / Tv_0)**(g/(Rd*gamma))

    p_tmp = p0 * (Tv/Tv_0)**(g/(Rd*gamma)) \
          * np.exp( -( (g*(z-z_trop)) / (Rd*Tv) ) )

    p[i_above_zt] = p_tmp[i_above_zt]

    return p, q, T


p_lay, q, T_lay = calc_p_q_T( z)
p_lev, _, T_lev = calc_p_q_T(zh)

# convert water from q to vmr
Rd_Rv = 287.04 / 461.5

# Again, I remove the vmr conversion to match Robert's case
h2o = q #/ (Rd_Rv * (1. - q))

co2 =  348.e-6
ch4 = 1650.e-9
n2o =  306.e-9
n2 = 0.7808
o2 = 0.2095

g1 = 3.6478
g2 = 0.83209
g3 = 11.3515
p_hpa = p_lay/100.

o3_min = 1e-13 # RRTMGP in Single Precision will fail with lower ozone concentrations
o3 = np.maximum(o3_min, g1 * p_hpa**g2 * np.exp(-p_hpa/g3) * 1e-6)

nc_file.createDimension("x", n_col_x)
nc_file.createDimension("y", n_col_y)
nc_file.createDimension("lay", p_lay.size)
nc_file.createDimension("lev", p_lev.size)
nc_file.createDimension("band_lw", n_bnd_lw)
nc_file.createDimension("band_sw", n_bnd_sw)

nc_z_lay = nc_file.createVariable("z_lay", float_type, ("lay"))
nc_z_lev = nc_file.createVariable("z_lev", float_type, ("lev"))
nc_z_lay[:] = z [:]
nc_z_lev[:] = zh[:]

nc_p_lay = nc_file.createVariable("p_lay", float_type, ("lay", "y", "x"))
nc_p_lev = nc_file.createVariable("p_lev", float_type, ("lev", "y", "x"))
nc_p_lay[:,:,:] = np.tile(p_lay[:,None, None], (1, n_col_y, n_col_x))
nc_p_lev[:,:,:] = np.tile(p_lev[:,None, None], (1, n_col_y, n_col_x))

nc_T_lay = nc_file.createVariable("t_lay", float_type, ("lay", "y", "x"))
nc_T_lev = nc_file.createVariable("t_lev", float_type, ("lev", "y", "x"))
nc_T_lay[:,:,:] = np.tile(T_lay[:,None, None], (1, n_col_y, n_col_x))
nc_T_lev[:,:,:] = np.tile(T_lev[:,None, None], (1, n_col_y, n_col_x))

nc_CO2 = nc_file.createVariable("vmr_co2", float_type)
nc_CH4 = nc_file.createVariable("vmr_ch4", float_type)
nc_N2O = nc_file.createVariable("vmr_n2o", float_type)
nc_O3  = nc_file.createVariable("vmr_o3" , float_type, ("lay", "y", "x"))
nc_H2O = nc_file.createVariable("vmr_h2o", float_type, ("lay", "y", "x"))
nc_N2  = nc_file.createVariable("vmr_n2" , float_type)
nc_O2  = nc_file.createVariable("vmr_o2" , float_type)

nc_CO2[:] = co2
nc_CH4[:] = ch4
nc_N2O[:] = n2o
nc_O3 [:,:,:] = np.tile(o3 [:,None, None], (1, n_col_y, n_col_x))
nc_H2O[:,:,:] = np.tile(h2o[:,None, None], (1, n_col_y, n_col_x))
nc_N2 [:] = n2
nc_O2 [:] = o2

# Longwave boundary conditions.
nc_emis_sfc = nc_file.createVariable("emis_sfc" , float_type, ("y", "x", "band_lw"))
nc_t_sfc = nc_file.createVariable("t_sfc" , float_type, ("y", "x"))

emis_sfc = 0.98
t_sfc = 300.

nc_emis_sfc[:,:,:] = emis_sfc
nc_t_sfc[:,:] = t_sfc

# Shortwave boundary conditions.
mu0 = 0.86

sfc_alb_dir = np.ones((n_col_y, n_col_x, n_bnd_sw))*0.06
sfc_alb_dif = np.ones((n_col_y, n_col_x, n_bnd_sw))*0.06

# total_solar_irradiance = 551.58

nc_mu0 = nc_file.createVariable("mu0", float_type, ("y", "x"))
nc_sfc_alb_dir = nc_file.createVariable("sfc_alb_dir", float_type, ("y", "x", "band_sw"))
nc_sfc_alb_dif = nc_file.createVariable("sfc_alb_dif", float_type, ("y", "x", "band_sw"))
# nc_tsi = nc_file.createVariable("tsi", float_type, ("y", "x"))

nc_mu0[:,:] = mu0
nc_sfc_alb_dir[:,:,:] = sfc_alb_dir[:,:]
nc_sfc_alb_dif[:,:,:] = sfc_alb_dif[:,:]
# nc_tsi[:,:] = total_solar_irradiance

# Clouds
nc_lwp = nc_file.createVariable('lwp', float_type, ('lay', 'y', 'x'))
nc_iwp = nc_file.createVariable('iwp', float_type, ('lay', 'y', 'x'))
nc_rel = nc_file.createVariable('rel', float_type, ('lay', 'y', 'x'))
nc_dei = nc_file.createVariable('dei', float_type, ('lay', 'y', 'x'))

min_rel, max_rel = 2.5, 21.5
min_dei, max_dei = 10., 180.

rel_val = 0.5*(min_rel + max_rel)
dei_val = 0.5*(min_dei + max_dei)

cloud_flag = (np.arange(1, n_col_x+1)%3 > 0)
cloud_mask = np.where((nc_p_lay[:,:,:] > 1.e4) & (nc_p_lay[:,:,:] < 9.e4) & cloud_flag[None, None,:], True, False)

nc_lwp[:,:,:] = np.where(cloud_mask & (nc_T_lay[:,:,:] > 263.), 10., 0.)
nc_iwp[:,:,:] = np.where(cloud_mask & (nc_T_lay[:,:,:] < 273.), 10., 0.)
nc_rel[:,:,:] = np.where(nc_lwp[:,:,:] > 0., rel_val, 0.)
nc_dei[:,:,:] = np.where(nc_iwp[:,:,:] > 0., dei_val, 0.)

nc_file.close()
