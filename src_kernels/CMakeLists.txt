#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles
    "../rte-rrtmgp/rte-frontend/mo_rte_kind.F90"
    "../rte-rrtmgp/rrtmgp-kernels/mo_gas_optics_rrtmgp_kernels.F90"
    "../src_kernels/mo_gas_optics_rrtmgp_kernels.F90"
    #"../rte-rrtmgp/rte-kernels/mo_rte_util_array.F90"
    "../rte-rrtmgp/rte-kernels/mo_optical_props_kernels.F90"
    "../rte-rrtmgp/rte-kernels/mo_rte_solver_kernels.F90"
    "../rte-rrtmgp/rte-kernels/mo_fluxes_broadband_kernels.F90"
    #"../src_kernels/mo_rte_solver_kernels.F90"

    # We use a slightly modified byband solver.
    "../src_kernels/mo_fluxes_byband_kernels.F90"
    "../src_kernels/mo_rte_util_array.F90"
    )

add_library(rte_rrtmgp_kernels STATIC ${sourcefiles})
