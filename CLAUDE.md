# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

C++ implementation of RTE+RRTMGP (Radiative Transfer for Energetics + Rapid Radiative Transfer Model for GCM applications Parallel), including a CUDA Monte Carlo ray tracer for 3D radiative transfer. Original Fortran version lives in the `rte-rrtmgp/` submodule.

## Build Commands

```bash
# CPU-only build (from repo root)
mkdir build && cd build
cmake -DSYST=macbook_brew ..
make

# CUDA build
cmake -DSYST=<config> -DUSECUDA=ON ..

# Single precision build
cmake -DSYST=<config> -DUSESP=ON ..

# Debug build
cmake -DSYST=<config> -DCMAKE_BUILD_TYPE=DEBUG ..
```

Config files are in `config/` (e.g., `macbook_brew`, `macbook_brew_gcc`, `ubuntu_22lts`).

## Running Tests

The build produces `test_rte_rrtmgp` (CPU) and optionally `test_rte_rrtmgp_gpu`, `test_rte_rrtmgp_rt_gpu`, `test_rte_rrtmgp_bw_gpu` (CUDA).

Test cases use Python scripts for setup, execution, and validation:

```bash
# RFMIP test (clear-sky reference case)
cd rfmip && ./make_links.sh && ln -sf ../build/test_rte_rrtmgp .
python3 rfmip_init.py && python3 rfmip_run.py
python3 compare-to-reference.py --ref_dir ../rrtmgp-data/examples/rfmip-clear-sky/reference --tst_dir . --var rld rlu rsd rsu --file 'r??_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc' --failure_threshold=5.8e-2

# All-sky test (cloudy conditions)
cd allsky && ./make_links.sh && ln -sf ../build/test_rte_rrtmgp .
python3 allsky_init.py && python3 allsky_run.py
python3 allsky_check.py --failure_threshold=5.8e-2
```

Required data files (linked via `make_links.sh`): `coefficients_lw.nc`, `coefficients_sw.nc`, `cloud_coefficients_lw.nc`, `cloud_coefficients_sw.nc`, and `rte_rrtmgp_input.nc`.

## Dependencies

- C++ and Fortran compilers (Fortran needed for `src_kernels/`)
- NetCDF (with HDF5/SZIP)
- Boost
- CUDA toolkit + cuRAND (optional, for GPU builds)
- Python 3 with numpy, netcdf4, xarray (for test scripts)
- Git submodules: `rte-rrtmgp/` (Fortran reference) and `rrtmgp-data/` (coefficient data)

## Architecture

### Dual CPU/GPU implementation pattern

Core physics classes exist in both CPU (`src/`, `include/`) and CUDA (`src_cuda/`, `include/`) variants. The CUDA ray tracer lives in `src_cuda_rt/` and `include_rt/`. Each CPU class has a GPU mirror with device-side memory management.

### Key classes

- **`Array<T,N>`** (`include/Array.h`): Template N-dimensional array, used throughout for all data storage
- **`Gas_optics_rrtmgp`**: Computes gas optical properties from k-distributions (largest source file)
- **`Optical_props`** / `Optical_props_1scl` / `Optical_props_2str`: Optical properties with 1-scalar or 2-stream representations
- **`Rte_lw`** / **`Rte_sw`**: Longwave and shortwave RTE solvers
- **`Cloud_optics`** / **`Aerosol_optics`**: Cloud and aerosol optical property parameterizations
- **`Source_functions`**: Planck source function calculations
- **`Gas_concs`**: Gas concentration container (maps gas names to arrays)
- **`Radiation_solver`** (`src_test/`): Top-level solver that orchestrates the full radiation calculation

### Precision and type abstraction

`include/types.h` defines `Float` (double or float via `-DRTE_USE_SP`), `Bool` (int or signed char via `-DRTE_USE_CBOOL`), and `Int` (unsigned long long). All physics code uses these type aliases.

### Fortran kernel bridge

`src_kernels/` contains Fortran kernels from the original RTE-RRTMGP, compiled and linked into the C++ library. Headers in `include/` declare the C-compatible interfaces (e.g., `rrtmgp_kernels.h`).

### Ray tracer (CUDA only)

Monte Carlo ray tracer in `src_cuda_rt/` and `include_rt/` supports forward (`Raytracer`), longwave (`Raytracer_lw`), and backward/adjoint (`Raytracer_bw`) modes. Uses cuRAND for random number generation. Helper functions for scattering (Rayleigh, Henyey-Greenstein) are in `raytracer_functions.h`.

### Test executables

`src_test/` contains the test driver code. `Radiation_solver.cpp` (CPU) and `Radiation_solver.cu` (GPU) implement the full solver pipeline used by all test cases. The different `.cu` test files correspond to different GPU executables (two-stream, ray tracer, backward).

## Licensing

Library code (`src/`, `include/`): BSD 3-clause. Test code (`src_test/`, `include_test/`): GPLv3.
