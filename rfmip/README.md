This is the RFMIP reference case as contained in the main RTE+RRTMGP repository.
In order to run the test, copy the `test_rte_rrtmgp` executable and the coefficient
files into this directory as `coefficients_lw.nc` and `coefficients_sw.nc`.

Follow the steps:

1. `./make_links.sh`                (link the executable, coefficients, and Python scripts)
2. `python stage_files.py`          (downloading the reference data)
3. `python rfmip_init.py`           (preparing the model input data)
4. `python rfmip_run.py`            (run the 1800 cases)
5. `python compare-to_reference.py` (compare output to reference file)
6. `python rfmip_plot.py`           (plot the cases in a colormesh per flux)

