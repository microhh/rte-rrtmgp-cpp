# C++/CUDA implementation of RTE+RRTMGP including ray tracer.

![Current build status](https://github.com/earth-system-radiation/rte-rrtmgp-cpp/actions/workflows/continuous-integration.yml/badge.svg?branch=main)

This is a C++ implementation (including a Monte Carlo ray tracer) of the Radiative Transfer for Energetics (RTE)
and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).

The original code is found at https://github.com/earth-system-radiation/rte-rrtmgp.

Contacts: Robert Pincus and Eli Mlawer
email: rrtmgp@aer.com

This C++ implementation can be downloaded from https://github.com/microhh/rte-rrtmgp-cpp

Contacts: Chiel van Heerwaarden and Menno Veerman

email: chiel.vanheerwaarden@wur.nl (questions on the C++ implementation)

email: menno.veerman@wur.nl (questions on the ray tracer)

Use and duplication is permitted under the terms of the
BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause

The source code of the testing executable in the `src_test` and
`include_test` directory is released under the GPLv3 license,
see https://www.gnu.org/licenses/gpl-3.0.en.html

In order to check out the code including the `rte-rrtmgp` submodule, use:

    git clone --recurse-submodules https://github.com/microhh/rte-rrtmgp-cpp

In case you had already checked out the repository, use:

    git submodule update --init


# Basic instructions
For building from source, create a build directory, for instance `build`.
From `build`, trigger `cmake .. -DSYST=config_file`, where `config_file` should be replaced by one of the configuration files in the `config` folder, for instance `-DSYST=macbook_brew`.

Building the source creates an executable `test_rte_rrtmgp`.
Three test cases are provided in directories `rfmip`, `allsky`, and `rcemip`.
In order to run those cases follow the instructions in the `README.md` of those respective directories.

In general, in order to run a test case, make sure the following files are present in the
directory from which `test_rte_rrtmgp` is triggered:
1. Input file `rte_rrtmgp_input.nc` with atmospheric profiles of pressure, temperature, and gases.
2. Long wave coefficients file from original RTE+RRTMGP repository (in `rrtmgp/data`) as `coefficients_lw.nc`
3. Short wave coefficients file from original RTE+RRTMGP repository (in `rrtmgp/data`) as `coefficients_sw.nc`
4. Long wave cloud optics coefficients file from original RTE+RRTMGP repository (in `rrtmgp-data`) as `cloud_coefficients_lw.nc`
5. Short wave cloud optics coefficients file from original RTE+RRTMGP repository (in `rrtmgp-data`) as `cloud_coefficients_sw.nc`
