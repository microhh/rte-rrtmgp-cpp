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

set(CUB_INCLUDE_DIR "/usr/local/cuda/include")
set(NETCDF_INCLUDE_DIR "/usr/include")
set(NETCDF_LIB_C       "/usr/lib/x86_64-linux-gnu/libnetcdf.so")
set(HDF5_LIB_1         "/usr/lib/x86_64-linux-gnu/libhdf5_serial.so")
set(HDF5_LIB_2         "/usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so")
set(SZIP_LIB           "")
set(LIBS ${NETCDF_LIB_C} ${HDF5_LIB_2} ${HDF5_LIB_1} ${SZIP_LIB} m z curl)
set(INCLUDE_DIRS ${NETCDF_INCLUDE_DIR} ${CUB_INCLUDE_DIR})

if(USECUDA)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CMAKE_CUDA_ARCHITECTURES 86)
  set(USER_CUDA_FLAGS "-std=c++17 -expt-relaxed-constexpr")
  set(USER_CUDA_FLAGS_RELEASE "-Xptxas -O3 -DNDEBUG")
  set(USER_CUDA_FLAGS_DEBUG "-Xptxas -O0 -g -G -DCUDACHECKS")
  # add_definitions(-DRTE_RRTMGP_GPU_MEMPOOL_OWN)
  add_definitions(-DRTE_RRTMGP_GPU_MEMPOOL_CUDA)
endif()

add_definitions(-DRTE_USE_CBOOL)
