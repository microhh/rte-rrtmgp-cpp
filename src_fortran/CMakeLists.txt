#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles
    "../src_fortran/mo_rte_kind.F90"
    "../rte-rrtmgp/rte/mo_rte_util_array.F90"
    "../rte-rrtmgp/rrtmgp/kernels/mo_gas_optics_kernels.F90"
    "../rte-rrtmgp/rrtmgp/kernels/mo_rrtmgp_util_reorder_kernels.F90"
    "../rte-rrtmgp/rte/kernels/mo_optical_props_kernels.F90"
    "../rte-rrtmgp/rte/kernels/mo_rte_solver_kernels.F90"
    "../rte-rrtmgp/rte/kernels/mo_fluxes_broadband_kernels.F90"
    "../rte-rrtmgp/extensions/mo_fluxes_byband_kernels.F90")

if(FLOAT_TYPE STREQUAL "single")
    message(STATUS "Compiling RRTMGP kernels in single precision")
    add_library(rte_rrtmgp_kernels STATIC ${sourcefiles})
    target_compile_definitions(rte_rrtmgp_kernels PRIVATE REAL_TYPE=sp)
else()
    message(STATUS "Compiling RRTMGP kernels in double precision")
    add_library(rte_rrtmgp_kernels STATIC ${sourcefiles})
    target_compile_definitions(rte_rrtmgp_kernels PRIVATE REAL_TYPE=dp)
endif()
