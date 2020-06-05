import numpy as np
import netCDF4 as nc

float_type = "f8"
n_col = 32**2
n_bnd_lw = 16
n_bnd_sw = 14

nc_file = nc.Dataset("rte_rrtmgp_input.nc", mode="w", datamodel="NETCDF4", clobber=True)

# Radiation profiles.
z_top = 100.e3
dz = 500.
z  = np.arange(dz/2, z_top, dz)
zh = np.arange(   0, z_top-dz/2, dz)
zh = np.append(zh, z_top)

def calc_p_q_T(z):
    q_0 = 0.01864 # for 300 K SST.
    z_q1 = 4.0e3
    z_q2 = 7.5e3
    z_t = 15.e3
    q_t = 1.e-14

    q = q_0 * np.exp(-z/z_q1) * np.exp(-(z/z_q2)**2)

    # CvH hack to remove moisture jump.
    q_t = q_0 * np.exp(-z_t/z_q1) * np.exp(-(z_t/z_q2)**2)

    i_above_zt = np.where(z > z_t)
    q[i_above_zt] = q_t
    
    T_0 = 300.
    gamma = 6.7e-3
    Tv_0 = (1. + 0.608*q_0)*T_0
    Tv = Tv_0 - gamma*z
    Tv_t = Tv_0 - gamma*z_t
    Tv[i_above_zt] = Tv_t
    T = Tv / (1. + 0.608*q)
    
    g = 9.79764
    Rd = 287.04
    p0 = 101480.

    p = p0 * (Tv / Tv_0)**(g/(Rd*gamma))
    
    p_tmp = p0 * (Tv_t/Tv_0)**(g/(Rd*gamma)) \
          * np.exp( -( (g*(z-z_t)) / (Rd*Tv_t) ) )
    
    p[i_above_zt] = p_tmp[i_above_zt]

    return p, q, T

p_lay, q, T_lay = calc_p_q_T( z)
p_lev, _, T_lev = calc_p_q_T(zh)

# convert water from q to vmr
Rd_Rv = 287.04 / 461.5
h2o = q / (Rd_Rv * (1. - q))

co2 =  348.e-6
ch4 = 1650.e-9
n2o =  306.e-9
n2 = 0.7808
o2 = 0.2095

g1 = 3.6478
g2 = 0.83209
g3 = 11.3515
p_hpa = p_lay/100.
o3 = g1 * p_hpa**g2 * np.exp(-p_hpa/g3) * 1e-6

nc_file.createDimension("col", n_col)
nc_file.createDimension("lay", p_lay.size)
nc_file.createDimension("lev", p_lev.size)
nc_file.createDimension("band_lw", n_bnd_lw)
nc_file.createDimension("band_sw", n_bnd_sw)

nc_z_lay = nc_file.createVariable("z_lay", float_type, ("lay"))
nc_z_lev = nc_file.createVariable("z_lev", float_type, ("lev"))
nc_z_lay[:] = z [:]
nc_z_lev[:] = zh[:]

nc_p_lay = nc_file.createVariable("p_lay", float_type, ("lay", "col"))
nc_p_lev = nc_file.createVariable("p_lev", float_type, ("lev", "col"))
nc_p_lay[:,:] = np.tile(p_lay[:,None], (1, n_col))
nc_p_lev[:,:] = np.tile(p_lev[:,None], (1, n_col))

nc_T_lay = nc_file.createVariable("t_lay", float_type, ("lay", "col"))
nc_T_lev = nc_file.createVariable("t_lev", float_type, ("lev", "col"))
nc_T_lay[:,:] = np.tile(T_lay[:,None], (1, n_col))
nc_T_lev[:,:] = np.tile(T_lev[:,None], (1, n_col))

nc_CO2 = nc_file.createVariable("vmr_co2", float_type)
nc_CH4 = nc_file.createVariable("vmr_ch4", float_type)
nc_N2O = nc_file.createVariable("vmr_n2o", float_type)
nc_O3  = nc_file.createVariable("vmr_o3" , float_type, ("lay", "col"))
nc_H2O = nc_file.createVariable("vmr_h2o", float_type, ("lay", "col"))
nc_N2  = nc_file.createVariable("vmr_n2" , float_type)
nc_O2  = nc_file.createVariable("vmr_o2" , float_type)

nc_CO2[:] = co2
nc_CH4[:] = ch4
nc_N2O[:] = n2o
nc_O3 [:,:] = np.tile(o3 [:,None], (1, n_col))
nc_H2O[:,:] = np.tile(h2o[:,None], (1, n_col))
nc_N2 [:] = n2
nc_O2 [:] = o2

# Longwave boundary conditions.
nc_emis_sfc = nc_file.createVariable("emis_sfc" , float_type, ("col", "band_lw"))
nc_t_sfc = nc_file.createVariable("t_sfc" , float_type, ("col"))

emis_sfc = 1.
t_sfc = 300.

nc_emis_sfc[:,:] = emis_sfc
nc_t_sfc[:] = t_sfc

# Shortwave boundary conditions.
solar_zenith_angle = np.deg2rad(42.05);
mu0 = np.cos(solar_zenith_angle)

sfc_alb_dir = np.ones((n_col, n_bnd_sw))*0.07
sfc_alb_dif = np.ones((n_col, n_bnd_sw))*0.07

total_solar_irradiance = 551.58

nc_mu0 = nc_file.createVariable("mu0", float_type, ("col"))
nc_sfc_alb_dir = nc_file.createVariable("sfc_alb_dir", float_type, ("col", "band_sw"))
nc_sfc_alb_dif = nc_file.createVariable("sfc_alb_dif", float_type, ("col", "band_sw"))
nc_tsi = nc_file.createVariable("tsi", float_type, ("col"))

nc_mu0[:] = mu0
nc_sfc_alb_dir[:,:] = sfc_alb_dir[:,:]
nc_sfc_alb_dif[:,:] = sfc_alb_dif[:,:]
nc_tsi[:] = total_solar_irradiance

nc_file.close()
