import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


expt = 0 
site = 0

nc_file_run = nc.Dataset('rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='r')
nc_file_ref = nc.Dataset('reference/rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='r')

shape_rld = nc_file_run.variables['rld'][:].shape
rld_run_2d = nc_file_run.variables['rld'][:, :, :].reshape( (shape_rld[0]*shape_rld[1], shape_rld[2]) ).transpose()
rld_ref_2d = nc_file_ref.variables['rld'][:, :, :].reshape( (shape_rld[0]*shape_rld[1], shape_rld[2]) ).transpose()

plt.figure(figsize=(10,6))
plt.subplot(311)
plt.pcolormesh(rld_run_2d)
plt.colorbar()
plt.title('rld')
plt.subplot(312)
plt.pcolormesh(rld_ref_2d)
plt.colorbar()
plt.subplot(313)
plt.pcolormesh(rld_run_2d - rld_ref_2d)
plt.tight_layout()
plt.colorbar()
plt.tight_layout()
plt.show()


nc_file_run = nc.Dataset('rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='r')
nc_file_ref = nc.Dataset('reference/rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='r')

shape_rlu = nc_file_run.variables['rlu'][:].shape
rlu_run_2d = nc_file_run.variables['rlu'][:, :, :].reshape( (shape_rlu[0]*shape_rlu[1], shape_rlu[2]) ).transpose()
rlu_ref_2d = nc_file_ref.variables['rlu'][:, :, :].reshape( (shape_rlu[0]*shape_rlu[1], shape_rlu[2]) ).transpose()

plt.figure(figsize=(10,6))
plt.subplot(311)
plt.pcolormesh(rlu_run_2d)
plt.colorbar()
plt.title('rlu')
plt.subplot(312)
plt.pcolormesh(rlu_ref_2d)
plt.colorbar()
plt.subplot(313)
plt.pcolormesh(rlu_run_2d - rlu_ref_2d)
plt.tight_layout()
plt.colorbar()
plt.tight_layout()
plt.show()


nc_file_run = nc.Dataset('rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='r')
nc_file_ref = nc.Dataset('reference/rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='r')

shape_rsd = nc_file_run.variables['rsd'][:].shape
rsd_run_2d = nc_file_run.variables['rsd'][:, :, :].reshape( (shape_rsd[0]*shape_rsd[1], shape_rsd[2]) ).transpose()
rsd_ref_2d = nc_file_ref.variables['rsd'][:, :, :].reshape( (shape_rsd[0]*shape_rsd[1], shape_rsd[2]) ).transpose()

plt.figure(figsize=(10,6))
plt.subplot(311)
plt.pcolormesh(rsd_run_2d)
plt.colorbar()
plt.title('rsd')
plt.subplot(312)
plt.pcolormesh(rsd_ref_2d)
plt.colorbar()
plt.subplot(313)
plt.pcolormesh(rsd_run_2d - rsd_ref_2d)
plt.tight_layout()
plt.colorbar()
plt.tight_layout()
plt.show()


nc_file_run = nc.Dataset('rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='r')
nc_file_ref = nc.Dataset('reference/rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='r')

shape_rsu = nc_file_run.variables['rsu'][:].shape
rsu_run_2d = nc_file_run.variables['rsu'][:, :, :].reshape( (shape_rsu[0]*shape_rsu[1], shape_rsu[2]) ).transpose()
rsu_ref_2d = nc_file_ref.variables['rsu'][:, :, :].reshape( (shape_rsu[0]*shape_rsu[1], shape_rsu[2]) ).transpose()

plt.figure(figsize=(10,6))
plt.subplot(311)
plt.pcolormesh(rsu_run_2d)
plt.colorbar()
plt.title('rsu')
plt.subplot(312)
plt.pcolormesh(rsu_ref_2d)
plt.colorbar()
plt.subplot(313)
plt.pcolormesh(rsu_run_2d - rsu_ref_2d)
plt.tight_layout()
plt.colorbar()
plt.tight_layout()
plt.show()


"""
plev_run = nc_file_run.variables['plev'][expt, :] / 1e3
plev_ref = nc_file_ref.variables['plev'][expt, :] / 1e3

rld_run = nc_file_run.variables['rld'][expt, site, :]
rld_ref = nc_file_ref.variables['rld'][expt, site, :]

plt.figure()
plt.subplot(121)
plt.plot(rld_run, plev_run, label='run')
plt.plot(rld_ref, plev_ref, label='ref')
plt.gca().invert_yaxis()
plt.xlabel('flux (W m-2)')
plt.ylabel('p (kPa)')
plt.legend(loc=0, frameon=False)
plt.subplot(122)
plt.plot(rld_run - rld_ref, plev_ref)
plt.gca().invert_yaxis()
plt.xlabel('d flux (W m-2)')
plt.ylabel('p (kPa)')
plt.tight_layout()
"""
