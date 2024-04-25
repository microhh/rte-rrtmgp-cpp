This is the allsky reference case as contained in the main RTE+RRTMGP repository.
In order to run the test, copy the `test_rte_rrtmgp` executable and the coefficient
files into this directory as `coefficients_lw.nc`, `coefficients_sw.nc`,
`cloud_coefficients_lw.nc`, `cloud_coefficients_sw.nc`.

Follow the steps:

1. `./make_links.sh`                                           (link the coefficients, and Python scripts)
2. `ln -s {BUILD_DIRECTORY}/{EXECUTABLE_NAME} test_rte_rrtmgp` (link the executable)
3. `python allsky_init.py`                                     (preparing the model input data)
4. `python allsky_run.py`                                      (run the model and prepare the file for comparison)
5. `python allsky_check.py`                                    (download reference file and compare output)
6. `python allsky_plot.py`                                     (plot the fluxes for clear and cloudy skies)

