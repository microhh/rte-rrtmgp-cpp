# Ubuntu 20.04
if(USEMPI) 
  set(ENV{CC}  mpicc ) # C compiler for parallel build
  set(ENV{CXX} mpicxx) # C++ compiler for parallel build
else()
  set(ENV{CC}  gcc) # C compiler for serial build
  set(ENV{CXX} g++) # C++ compiler for serial build
endif()

set(USER_CXX_FLAGS "-std=c++17")
set(USER_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
set(USER_FC_FLAGS "-fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_FC_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(CURAND_LIB_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64")
set(CURAND_INCLUDE_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/include")
set(CURAND_LIB "${CURAND_LIB_DIR}/libcurand.so")
set(NETCDF_INCLUDE_DIR "/opt/cray/pe/netcdf/4.9.0.9/gnu/12.3/include")
set(NETCDF_LIB_C "/opt/cray/pe/netcdf/4.9.0.9/gnu/12.3/lib/libnetcdf.so")
set(HDF5_LIB_1 "/opt/cray/pe/hdf5/1.12.2.9/gnu/12.3/lib/libhdf5.so")        # Main HDF5 library
set(HDF5_LIB_2 "/opt/cray/pe/hdf5/1.12.2.9/gnu/12.3/lib/libhdf5_hl.so")     # High-level HDF5 library


set(SZIP_LIB           "")
set(LIBS ${NETCDF_LIB_C} ${HDF5_LIB_2} ${HDF5_LIB_1} ${CURAND_LIB} ${SZIP_LIB} m z curl)
set(INCLUDE_DIRS ${NETCDF_INCLUDE_DIR} ${CURAND_INCLUDE_DIR})

if(USECUDA)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CMAKE_CUDA_ARCHITECTURES 80)
  set(USER_CUDA_FLAGS "-std=c++17 -expt-relaxed-constexpr")
  set(USER_CUDA_FLAGS_RELEASE "-Xptxas -O3 -DNDEBUG")
  set(USER_CUDA_FLAGS_DEBUG "-Xptxas -O0 -g -G -DCUDACHECKS")
  # add_definitions(-DRTE_RRTMGP_GPU_MEMPOOL_OWN)
  add_definitions(-DRTE_RRTMGP_GPU_MEMPOOL_CUDA)
endif()

add_definitions(-DRTE_USE_CBOOL)
