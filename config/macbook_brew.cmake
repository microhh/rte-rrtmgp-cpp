# MacBook
if(USEMPI) 
  set(ENV{CC}  mpicc ) # C compiler for parallel build
  set(ENV{CXX} mpicxx) # C++ compiler for parallel build
  set(ENV{FC}  mpif90) # Fortran compiler for parallel build
else()
  set(ENV{CC}  clang   ) # C compiler for serial build
  set(ENV{CXX} clang++ ) # C++ compiler for serial build
  set(ENV{FC}  gfortran) # Fortran compiler for parallel build
endif()

set(GNU_SED "gsed")

set(USER_CXX_FLAGS "-std=c++14")
set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
set(USER_FC_FLAGS "-std=f2003 -fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_FC_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(NETCDF_INCLUDE_DIR "/usr/local/include")
set(NETCDF_LIB_C       "/usr/local/lib/libnetcdf.dylib")
set(HDF5_LIB_1         "/usr/local/lib/libhdf5.dylib")
set(HDF5_LIB_2         "/usr/local/lib/libhdf5_hl.dylib")
set(SZIP_LIB           "/usr/local/lib/libsz.dylib")
set(LIBS ${NETCDF_LIB_CPP} ${NETCDF_LIB_C} ${HDF5_LIB_2} ${HDF5_LIB_1} ${SZIP_LIB} m z curl)
set(INCLUDE_DIRS ${FFTW_INCLUDE_DIR} ${NETCDF_INCLUDE_DIR})

add_definitions(-DRESTRICTKEYWORD=__restrict__)
add_definitions(-DUSE_CBOOL)
