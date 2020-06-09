# Instruction to the neural networks-based gas optics solver
1. *to be done* Generate atmospheric profiles of temperature, pressure, water vapour and ozone (optionally) that are represent the range of these variabiles in the case of interest.
2. *to be done* Run RRTMGP on these profiles to obtain training data
3. *to be done* Train neural networks and export weights to netCDF
4. copy netCDF file with weights to build directory and rename to "weights.nc"