# Annuna GPU nodes

# salloc -c 8 --gres=gpu:1 --constraint=V100 -t 1:00:00 --partition=gpu

# module load cuda
# module load netcdf/gcc/64/4.6.1
# module load fftw3/gcc/64/3.3.8
# module load hdf5/gcc/64/1.10.1
# module unload intel
# module load gcc/7.1.0
# module unload python
# module load python/3.7.1

set(ENV{CC}  gcc) # C compiler for serial build
set(ENV{CXX} g++) # C++ compiler for serial build

set(USER_CXX_FLAGS "-std=c++14")
set(USER_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
set(USER_FC_FLAGS "-std=f2003 -fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(USER_FC_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(NETCDF_LIB_C "netcdf")
set(HDF5_LIB     "hdf5")

set(LIBS ${NETCDF_LIB_C} ${HDF5_LIB} m z curl)

if(USECUDA)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUFFT_LIB "/cm/shared/apps/cuda/10.1/lib64/libcufft.so")
  set(LIBS ${LIBS} ${CUFFT_LIB} -rdynamic )
  set(USER_CUDA_NVCC_FLAGS "-arch=sm_70 -std=c++14")
  set(USER_CUDA_NVCC_FLAGS_RELEASE "-Xptxas -O3 -DNDEBUG")
endif()

add_definitions(-DRESTRICTKEYWORD=__restrict__)
add_definitions(-DUSE_CBOOL)
