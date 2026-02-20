import subprocess
import os
import netCDF4 as nc

def remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

remove('allsky_output.nc')
subprocess.run(['./test_rte_rrtmgp', 'allsky'])

remove('rrtmgp-allsky.nc')


in_file = nc.Dataset("allsky_output.nc")
out_file = nc.Dataset("rrtmgp-allsky.nc", "w")

for dim_name in ["band_sw", "band_lw", "gpt_sw", "gpt_lw", "pair", "lay", "lev"]:    
    out_file.createDimension(dim_name, in_file.dimensions[dim_name].size)
   
#merge x,y dimensions to col
out_file.createDimension("col", in_file.dimensions['x'].size * in_file.dimensions['y'].size)

for name, variable in in_file.variables.items():
    var_dims = variable.dimensions
    var_dims = tuple(list(var_dims)[:-2] + ["col"]) if ("x" in var_dims and "y" in var_dims) else var_dims
    name_out = "sw_flux_dir" if name == "sw_flux_dn_dir" else name
    tmp = out_file.createVariable(name_out, variable.datatype, var_dims)
    out_file.variables[name_out][:] = in_file.variables[name][:]

in_file.close()
out_file.close()
