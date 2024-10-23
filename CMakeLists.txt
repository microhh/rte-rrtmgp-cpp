#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
cmake_minimum_required (VERSION 3.18)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/config)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set the precision of the build.
if(USESP)
  message(STATUS "Precision: Single (32-bits floats)")
  add_compile_definitions(RTE_USE_SP)
else()
  message(STATUS "Precision: Double (64-bits floats)")
endif()

# Load system specific settings if not set, force default.cmake.
if(NOT SYST)
  set(SYST default)
endif()
include(${SYST} OPTIONAL RESULT_VARIABLE SYSTINC)

# Trigger fatal error if illegal module is loaded.
if(${SYSTINC} STREQUAL "NOTFOUND")
  message(FATAL_ERROR "Config file config/" ${SYST} ".cmake does not exist.")
endif()

# Set the default build type to RELEASE.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE CACHE STRING
    "Choose the type of build, options are: None Debug Release." FORCE)
else()
  string(TOUPPER ${CMAKE_BUILD_TYPE} TEMP)
  set(CMAKE_BUILD_TYPE ${TEMP} CACHE STRING
    "Choose the type of build, options are: None Debug Release." FORCE)
endif()

# Start the project only after the system specific settings are loaded.
if(USECUDA)
  project(rte-rrtmgp-cpp C CXX Fortran CUDA)
else()
  project(rte-rrtmgp-cpp C CXX Fortran)
endif()

# Load the CUDA module in case CUDA is enabled and display status message.
if(USECUDA)
  message(STATUS "CUDA: Enabled.")
  add_definitions("-DUSECUDA")
else()
  message(STATUS "CUDA: Disabled.")
endif()

# Only set the compiler flags when the cache is created
# to enable editing of the flags in the CMakeCache.txt file.
if(NOT HASCACHE)
  set(CMAKE_C_FLAGS ${USER_C_FLAGS} CACHE STRING
    "Flags used by the C-compiler during all build types." FORCE)
  set(CMAKE_C_FLAGS_DEBUG ${USER_C_FLAGS_DEBUG} CACHE STRING
    "Flags used by the C-compiler during debug builds." FORCE)
  set(CMAKE_C_FLAGS_RELEASE ${USER_C_FLAGS_RELEASE} CACHE STRING
    "Flags used by the C-compiler during release builds." FORCE)

  set(CMAKE_CXX_FLAGS ${USER_CXX_FLAGS} CACHE STRING
    "Flags used by the CXX-compiler during all build types." FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG ${USER_CXX_FLAGS_DEBUG} CACHE STRING
    "Flags used by the CXX-compiler during debug builds." FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE ${USER_CXX_FLAGS_RELEASE} CACHE STRING
    "Flags used by the CXX-compiler during release builds." FORCE)

  set(CMAKE_Fortran_FLAGS ${USER_FC_FLAGS} CACHE STRING
    "Flags used by the Fortran-compiler during all build types." FORCE)
  set(CMAKE_Fortran_FLAGS_DEBUG ${USER_FC_FLAGS_DEBUG} CACHE STRING
    "Flags used by the Fortran-compiler during debug builds." FORCE)
  set(CMAKE_Fortran_FLAGS_RELEASE ${USER_FC_FLAGS_RELEASE} CACHE STRING
    "Flags used by the Fortran-compiler during release builds." FORCE)

  if(USECUDA)
    set(CMAKE_CUDA_FLAGS ${USER_CUDA_FLAGS} CACHE STRING
      "Flags used by the CXX-compiler during all build types." FORCE)
    set(CMAKE_CUDA_FLAGS_DEBUG ${USER_CUDA_FLAGS_DEBUG} CACHE STRING
      "Flags used by the CXX-compiler during debug builds." FORCE)
    set(CMAKE_CUDA_FLAGS_RELEASE ${USER_CUDA_FLAGS_RELEASE} CACHE STRING
      "Flags used by the CXX-compiler during release builds." FORCE)
  endif()

  message(STATUS "Build Type: " ${CMAKE_BUILD_TYPE})
  set(HASCACHE TRUE CACHE BOOL "CMakeCache.txt created." FORCE)

  # Make sure that ccmake only contains build type.
  mark_as_advanced(HASCACHE)
  mark_as_advanced(CMAKE_INSTALL_PREFIX)
endif()

# Print the C++ and CUDA compiler flags to the screen.
if(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
  message(STATUS "CXX-compiler flags: " ${CMAKE_CXX_FLAGS} " " ${CMAKE_CXX_FLAGS_RELEASE})
  message(STATUS "Fortran-compiler flags: " ${CMAKE_CXX_FLAGS} " " ${CMAKE_CXX_FLAGS_RELEASE})
else()
  message(STATUS "CXX-compiler flags: " ${CMAKE_CXX_FLAGS} " " ${CMAKE_CXX_FLAGS_DEBUG})
  message(STATUS "Fortran-compiler flags: " ${CMAKE_CXX_FLAGS} " " ${CMAKE_CXX_FLAGS_DEBUG})
endif()
if(USECUDA)
  if(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    message(STATUS "NVCC-compiler flags: " ${CMAKE_CUDA_FLAGS} " " ${CMAKE_CUDA_FLAGS_RELEASE})
  else()
    message(STATUS "NVCC-compiler flags: " ${CMAKE_CUDA_FLAGS} " " ${CMAKE_CUDA_FLAGS_DEBUG})
  endif()
endif()

add_subdirectory(src_kernels)
add_subdirectory(src)
if(USECUDA)
  add_subdirectory(src_kernels_cuda)
  add_subdirectory(src_cuda)
  add_subdirectory(src_kernels_cuda_rt)
  add_subdirectory(src_cuda_rt)
endif()
add_subdirectory(src_test)
