import numpy as np
import netCDF4 as nc
import shutil
import subprocess


expts = 18


# Run the experiments.
for expt in range(expts):
    shutil.copyfile('rfmip_input_expt_{:02d}.nc'.format(expt), 'rfmip_input.nc')
    subprocess.run(['./test_rte_rrtmgp', 'rfmip'])
    shutil.move('rfmip_output.nc', 'rfmip_output_expt_{:02d}.nc'.format(expt))
    print(' ')


# Prepare the output file.
for expt in range(expts):
    # Save all the input data to NetCDF
    nc_file = nc.Dataset('rfmip_output_expt_{:02d}.nc'.format(expt), mode='r')
    
    nc_file_rld = nc.Dataset('rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
    nc_file_rlu = nc.Dataset('rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
    nc_file_rsd = nc.Dataset('rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
    nc_file_rsu = nc.Dataset('rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
    
    nc_file_rld.variables['rld'][expt,:,:] = nc_file.variables['lw_flux_dn'][:,:].swapaxes(0,2)
    nc_file_rlu.variables['rlu'][expt,:,:] = nc_file.variables['lw_flux_up'][:,:].swapaxes(0,2)
    nc_file_rsd.variables['rsd'][expt,:,:] = nc_file.variables['sw_flux_dn'][:,:].swapaxes(0,2)
    nc_file_rsu.variables['rsu'][expt,:,:] = nc_file.variables['sw_flux_up'][:,:].swapaxes(0,2)
    
    nc_file.close()
    nc_file_rld.close()
    nc_file_rlu.close()
    nc_file_rsd.close()
    nc_file_rsu.close()

