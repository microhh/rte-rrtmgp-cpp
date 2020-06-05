import numpy as np
import netCDF4 as nc

# Settings
float_type = 'f8'
ncol = 2

band_lw = 16
band_sw = 14

# Save all the input data to NetCDF
nc_file = nc.Dataset('rte_rrtmgp_input.nc', mode='w', datamodel='NETCDF4', clobber=True)
nc_file_garand = nc.Dataset('garand-atmos-1.nc', mode='r', datamodel='NETCDF4', clobber=False)

# Create a group for the radiation and set up the values.
nc_file.createDimension('lay', nc_file_garand.dimensions['lay'].size)
nc_file.createDimension('lev', nc_file_garand.dimensions['lev'].size)
nc_file.createDimension('col', ncol)
nc_file.createDimension('band_lw', band_lw)
nc_file.createDimension('band_sw', band_sw)

nc_p_lay = nc_file.createVariable('p_lay', float_type, ('lay', 'col'))
nc_p_lev = nc_file.createVariable('p_lev', float_type, ('lev', 'col'))
nc_t_lay = nc_file.createVariable('t_lay', float_type, ('lay', 'col'))
nc_t_lev = nc_file.createVariable('t_lev', float_type, ('lev', 'col'))

nc_p_lay[:,:] = nc_file_garand.variables['p_lay'][:,:]
nc_p_lev[:,:] = nc_file_garand.variables['p_lev'][:,:]

# Make sure the top edge does not exceed the minimum tolerable pressure
# of the coefficient files.
nc_p_lev[:,:] = np.maximum(nc_p_lev[:,:], np.nextafter(1.005183574463, 1.))

nc_t_lay[:,:] = nc_file_garand.variables['t_lay'][:,:]
nc_t_lev[:,:] = nc_file_garand.variables['t_lev'][:,:]

nc_col_dry = nc_file.createVariable('col_dry', float_type, ('lay', 'col'))
nc_col_dry[:,:] = nc_file_garand.variables['col_dry'][:,:]

nc_surface_emissivity = nc_file.createVariable('emis_sfc', float_type, ('col', 'band_lw'))
nc_surface_temperature = nc_file.createVariable('t_sfc', float_type, 'col')

nc_surface_emissivity[:,:] = 0.98
nc_surface_temperature[:] = nc_t_lev[0,:]

nc_surface_albedo_dir = nc_file.createVariable('sfc_alb_dir', float_type, ('col', 'band_sw'))
nc_surface_albedo_dif = nc_file.createVariable('sfc_alb_dif', float_type, ('col', 'band_sw'))

nc_surface_albedo_dir[:,:] = 0.06
nc_surface_albedo_dif[:,:] = 0.06

nc_mu0 = nc_file.createVariable('mu0', float_type, ('col'))
nc_mu0[:] = 0.86

nc_h2o     = nc_file.createVariable('vmr_h2o', float_type, ('lay', 'col'))
nc_co2     = nc_file.createVariable('vmr_co2', float_type, ('lay', 'col'))
nc_o3      = nc_file.createVariable('vmr_o3' , float_type, ('lay', 'col'))
nc_n2o     = nc_file.createVariable('vmr_n2o', float_type, ('lay', 'col'))
nc_co      = nc_file.createVariable('vmr_co' , float_type, ('lay', 'col'))
nc_ch4     = nc_file.createVariable('vmr_ch4', float_type, ('lay', 'col'))
nc_o2      = nc_file.createVariable('vmr_o2' , float_type, ('lay', 'col'))
nc_n2      = nc_file.createVariable('vmr_n2' , float_type, ('lay', 'col'))

# Profiles
nc_h2o[:,:] = nc_file_garand.variables['vmr_h2o'][:,:]
nc_co2[:,:] = nc_file_garand.variables['vmr_co2'][:,:]
nc_o3 [:,:] = nc_file_garand.variables['vmr_o3' ][:,:]
nc_n2o[:,:] = nc_file_garand.variables['vmr_n2o'][:,:]
nc_co [:,:] = nc_file_garand.variables['vmr_co' ][:,:]
nc_ch4[:,:] = nc_file_garand.variables['vmr_ch4'][:,:]
nc_o2 [:,:] = nc_file_garand.variables['vmr_o2' ][:,:]
nc_n2 [:,:] = nc_file_garand.variables['vmr_n2' ][:,:]

# Clouds
nc_lwp = nc_file.createVariable('lwp', float_type, ('lay', 'col'))
nc_iwp = nc_file.createVariable('iwp', float_type, ('lay', 'col'))
nc_rel = nc_file.createVariable('rel', float_type, ('lay', 'col'))
nc_rei = nc_file.createVariable('rei', float_type, ('lay', 'col'))

#rel_val = 0.5 * (cloud_optics%get_min_radius_liq() + cloud_optics%get_max_radius_liq())
#rei_val = 0.5 * (cloud_optics%get_min_radius_ice() + cloud_optics%get_max_radius_ice())
min_rel, max_rel = 2.5, 21.5
min_rei, max_rei = 10., 180.

rel_val = 0.5*(min_rel + max_rel)
rei_val = 0.5*(min_rei + max_rei)

nlay, ncol = nc_p_lay[:,:].shape[0], nc_p_lay[:,:].shape[1]
cloud_mask = np.zeros((nlay, ncol))

for ilay in range(nlay):
    for icol in range(ncol):
        cloud_mask[ilay, icol] = (nc_p_lay[ilay, icol] > 1.e4) and (nc_p_lay[ilay, icol] < 9.e4) and (icol%3 != 0)
        nc_lwp[ilay, icol] = 10. if (cloud_mask[ilay, icol] and (nc_t_lay[ilay, icol] > 263.)) else 0.
        nc_iwp[ilay, icol] = 10. if (cloud_mask[ilay, icol] and (nc_t_lay[ilay, icol] < 273.)) else 0.
        nc_rel[ilay, icol] = rel_val if cloud_mask[ilay, icol] else 0.
        nc_rei[ilay, icol] = rei_val if cloud_mask[ilay, icol] else 0.
 
nc_file_garand.close()
nc_file.close()