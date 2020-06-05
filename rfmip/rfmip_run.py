import numpy as np
import netCDF4 as nc
import shutil
import subprocess


expts = 18

# Run the experiments.
for expt in range(expts):
    shutil.copyfile('rte_rrtmgp_input_expt_{:02d}.nc'.format(expt), 'rte_rrtmgp_input.nc')
    subprocess.run(['./test_rte_rrtmgp'])
    shutil.move('rte_rrtmgp_output.nc', 'rte_rrtmgp_output_expt_{:02d}.nc'.format(expt))


# Prepare the output file.
for expt in range(expts):
    # Save all the input data to NetCDF
    nc_file = nc.Dataset('rte_rrtmgp_output_expt_{:02d}.nc'.format(expt), mode='r')
    
    nc_file_rld = nc.Dataset('reference/rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
    nc_file_rlu = nc.Dataset('reference/rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
    nc_file_rsd = nc.Dataset('reference/rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
    nc_file_rsu = nc.Dataset('reference/rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
    
    nc_file_rld.variables['rld'][expt,:,:] = nc_file.variables['lw_flux_dn'][:,:].transpose()
    nc_file_rlu.variables['rlu'][expt,:,:] = nc_file.variables['lw_flux_up'][:,:].transpose()
    nc_file_rsd.variables['rsd'][expt,:,:] = nc_file.variables['sw_flux_dn'][:,:].transpose()
    nc_file_rsu.variables['rsu'][expt,:,:] = nc_file.variables['sw_flux_up'][:,:].transpose()
    
    nc_file.close()
    nc_file_rld.close()
    nc_file_rlu.close()
    nc_file_rsd.close()
    nc_file_rsu.close()

