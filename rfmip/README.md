This is the RFMIP reference case as contained in the main RTE+RRTMGP repository.
In order to run the test, copy the `test_rte_rrtmgp` executable and the coefficient
files into this directory as `coefficients_lw.nc` and `coefficients_sw.nc`.

Follow the steps:

1. `./make_links.sh`                                           (link the coefficients, and Python scripts)
2. `ln -s {BUILD_DIRECTORY}/{EXECUTABLE_NAME} test_rte_rrtmgp` (link the executable)
3. `./check_rfmip.sh`                                          (check the outcomes)
4. `python rfmip_plot.py`                                      (plot the cases in a colormesh per flux)

