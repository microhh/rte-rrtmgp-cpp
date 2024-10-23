#! /bin/sh
ln -sf ../rrtmgp-data/rrtmgp-clouds-sw.nc cloud_coefficients_sw.nc
ln -sf ../rrtmgp-data/rrtmgp-clouds-lw.nc cloud_coefficients_lw.nc
ln -sf ../rrtmgp-data/rrtmgp-gas-sw-g224.nc coefficients_sw.nc
ln -sf ../rrtmgp-data/rrtmgp-gas-lw-g256.nc coefficients_lw.nc
echo "Don't forget to link your executable as 'test_rte_rrtmgp'"
