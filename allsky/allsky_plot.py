import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

nc_file_run = nc.Dataset('rrtmgp-allsky.nc', 'r')
nc_file_ref = nc.Dataset('ref/rrtmgp-allsky.nc', 'r')

p_lev = nc_file_run.variables['p_lev'][:,0] / 1e3
cols = np.arange(128)

sw_flux_dn_run = nc_file_run.variables['sw_flux_dn'][:,:]
sw_flux_dn_ref = nc_file_ref.variables['sw_flux_dn'][:,:]

sw_flux_dir_run = nc_file_run.variables['sw_flux_dir'][:,:]
sw_flux_dir_ref = nc_file_ref.variables['sw_flux_dir'][:,:]

lw_flux_dn_run = nc_file_run.variables['lw_flux_dn'][:,:]
lw_flux_dn_ref = nc_file_ref.variables['lw_flux_dn'][:,:]

sw_flux_up_run = nc_file_run.variables['sw_flux_up'][:,:]
sw_flux_up_ref = nc_file_ref.variables['sw_flux_up'][:,:]

lw_flux_up_run = nc_file_run.variables['lw_flux_up'][:,:]
lw_flux_up_ref = nc_file_ref.variables['lw_flux_up'][:,:]

plt.figure(figsize=(10,6))
plt.subplot(231)
plt.plot(sw_flux_dn_run[:,2], p_lev, 'C0-', label='clear')
plt.plot(sw_flux_dn_ref[:,2], p_lev, 'C0:')
plt.plot(sw_flux_dn_run[:,0], p_lev, 'C1-', label='cloud')
plt.plot(sw_flux_dn_ref[:,0], p_lev, 'C1:')
plt.gca().invert_yaxis()
plt.xlabel(r'sw_flux_dn (W m-2)')
plt.ylabel(r'p (kPa)')
plt.legend(loc=0, frameon=False)

plt.subplot(232)
plt.plot(sw_flux_dir_run[:,2], p_lev, 'C0-', label='clear')
plt.plot(sw_flux_dir_ref[:,2], p_lev, 'C0:')
plt.plot(sw_flux_dir_run[:,0], p_lev, 'C1-', label='cloud')
plt.plot(sw_flux_dir_ref[:,0], p_lev, 'C1:')
plt.gca().invert_yaxis()
plt.xlabel(r'sw_flux_dir (W m-2)')
plt.ylabel(r'p (kPa)')
plt.legend(loc=0, frameon=False)

plt.subplot(233)
plt.plot(sw_flux_up_run[:,2], p_lev, 'C0-', label='clear')
plt.plot(sw_flux_up_ref[:,2], p_lev, 'C0:')
plt.plot(sw_flux_up_run[:,0], p_lev, 'C1-', label='cloud')
plt.plot(sw_flux_up_ref[:,0], p_lev, 'C1:')
plt.gca().invert_yaxis()
plt.xlabel(r'sw_flux_up (W m-2)')
plt.ylabel(r'p (kPa)')
plt.legend(loc=0, frameon=False)

plt.subplot(234)
plt.plot(lw_flux_dn_run[:,2], p_lev, 'C0-', label='clear')
plt.plot(lw_flux_dn_ref[:,2], p_lev, 'C0:')
plt.plot(lw_flux_dn_run[:,0], p_lev, 'C1-', label='cloud')
plt.plot(lw_flux_dn_ref[:,0], p_lev, 'C1:')
plt.gca().invert_yaxis()
plt.xlabel(r'lw_flux_dn (W m-2)')
plt.ylabel(r'p (kPa)')
plt.legend(loc=0, frameon=False)

plt.subplot(236)
plt.plot(lw_flux_up_run[:,2], p_lev, 'C0-', label='clear')
plt.plot(lw_flux_up_ref[:,2], p_lev, 'C0:')
plt.plot(lw_flux_up_run[:,0], p_lev, 'C1-', label='cloud')
plt.plot(lw_flux_up_ref[:,0], p_lev, 'C1:')
plt.gca().invert_yaxis()
plt.xlabel(r'lw_flux_up (W m-2)')
plt.ylabel(r'p (kPa)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()

plt.show()
