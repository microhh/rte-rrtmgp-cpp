import subprocess
import os

def remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

remove('rte_rrtmgp_output.nc')
subprocess.run(['./test_rte_rrtmgp'])

remove('rrtmgp-allsky.nc')
subprocess.run('ncrename -h -v sw_flux_dn_dir,sw_flux_dir -d col,col_flx rte_rrtmgp_output.nc rrtmgp-allsky.nc'.split())

