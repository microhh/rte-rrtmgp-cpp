#! /bin/sh
python3 rfmip_init.py
python3 rfmip_run.py
python3 compare-to-reference.py --ref_dir ../rrtmgp-data/examples/rfmip-clear-sky/reference --tst_dir . \
--var rld rlu rsd rsu --file r??_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc --failure_threshold=7e-4

