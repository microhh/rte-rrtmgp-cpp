# Example LES cloudfield
Example cloudfield from a 2560 x 2560 x 4000 m3 (dx=dy=dz=20m) large-eddy simulations based on the Rain In Cumulus over the Ocean (RICO) campaign (Van Zanten, 2011).
Input and reference output files are stored on Zenodo: https://doi.org/10.5281/zenodo.18757088

How to run:

1. `Compile code (following the basic instructions) with -DUSECUDA`
2. `ln -s {BUILD_DIRECTORY}/{EXECUTABLE_NAME} test_rte_rrtmgp_*` (link the executables)
3. `./make_links.sh`                                             (link the coefficients)
4. `download input and reference output files from https://doi.org/10.5281/zenodo.18757088`
5. `run "./test_rte_rrtmgp_rt test" (forward) or "test_rte_rrtmgp_bw test" (backward)`
