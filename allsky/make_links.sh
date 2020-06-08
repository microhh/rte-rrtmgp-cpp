#! /bin/sh
ln -sf ../rte-rrtmgp/extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc cloud_coefficients_sw.nc
ln -sf ../rte-rrtmgp/extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-lw.nc cloud_coefficients_lw.nc
ln -sf ../rte-rrtmgp/rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc coefficients_sw.nc
ln -sf ../rte-rrtmgp/rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc coefficients_lw.nc
ln -sf ../rte-rrtmgp/examples/all-sky/garand-atmos-1.nc .
ln -sf ../rte-rrtmgp/examples/all-sky/compare-to-reference.py .
ln -sf ../build/test_rte_rrtmgp .
