#! /bin/sh
ln -sf ../rrtmgp-data/rrtmgp-clouds-sw.nc cloud_coefficients_sw.nc
ln -sf ../rrtmgp-data/rrtmgp-clouds-lw.nc cloud_coefficients_lw.nc
ln -sf ../rrtmgp-data/rrtmgp-gas-sw-g112.nc coefficients_sw.nc
ln -sf ../rrtmgp-data/rrtmgp-gas-lw-g128.nc coefficients_lw.nc
ln -sf ../data/aerosol_optics_sw.nc
ln -sf ../data/aerosol_optics_lw.nc

echo "Don't forget to link the raytracering executables 'test_rte_rrtmgp_rt' and 'test_rte_rrtmgp_bw'"
