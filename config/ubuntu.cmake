# Ubuntu 16.04
if(USEMPI) 
  set(ENV{CC}  mpicc ) # C compiler for parallel build
  set(ENV{CXX} mpicxx) # C++ compiler for parallel build
  set(ENV{FC}  mpif90) # Fortran compiler for parallel build
else()
  set(ENV{CC}  gcc) # C compiler for serial build
  set(ENV{CXX} g++) # C++ compiler for serial build
  set(ENV{FC}  gfortran) # Fortran compiler for serial build
endif()

set(USER_CXX_FLAGS "-std=c++14 -DBOOL_TYPE=\"signed char\"")
set(USER_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
set(USER_FC_FLAGS "-std=f2003 -fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(USER_FC_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(NETCDF_INCLUDE_DIR "/usr/include")
set(NETCDF_LIB_C       "/usr/lib/x86_64-linux-gnu/libnetcdf.so")
set(HDF5_LIB_1         "/usr/lib/x86_64-linux-gnu/libhdf5_serial.so")
set(HDF5_LIB_2         "/usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so")

set(LIBS ${NETCDF_LIB_C} ${HDF5_LIB_2} ${HDF5_LIB_1} m z curl)
set(INCLUDE_DIRS ${FFTW_INCLUDE_DIR} ${NETCDF_INCLUDE_DIR})

add_definitions(-DRESTRICTKEYWORD=__restrict__)
add_definitions(-DUSE_CBOOL)
