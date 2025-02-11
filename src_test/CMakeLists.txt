#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
include_directories(${INCLUDE_DIRS} "../include" "../include_test" "../include_rt" "../include_rt_kernels/" "../include_kernels_cuda")

# retrieve the git hash from the current commit
find_package(Git)
if(GIT_EXECUTABLE)
  exec_program(git ${CMAKE_CURRENT_SOURCE_DIR} ARGS "describe --dirty" OUTPUT_VARIABLE GITHASH RETURN_VALUE GITRETURN)
  if(NOT GITRETURN MATCHES 0)
    set(GITHASH "not available")
  endif()
else()
  set(GITHASH "not available")
endif()

message(STATUS "Git hash " ${GITHASH})

# send a precompiler statement replacing the git hash
add_definitions(-DGITHASH="${GITHASH}")

if(USECUDA)
  add_executable(test_rte_rrtmgp_gpu Radiation_solver.cu test_rte_rrtmgp.cu)
  target_link_libraries(test_rte_rrtmgp_gpu rte_rrtmgp rte_rrtmgp_cuda ${LIBS} m)
  
  add_executable(test_rte_rrtmgp_rt_gpu Radiation_solver_rt.cu test_rte_rrtmgp_rt.cu)
  target_link_libraries(test_rte_rrtmgp_rt_gpu rte_rrtmgp rte_rrtmgp_cuda rte_rrtmgp_cuda_rt curand ${LIBS} m)
  
  add_executable(test_rte_rrtmgp_bw_gpu Radiation_solver_bw.cu test_rte_rrtmgp_bw.cu)
  target_link_libraries(test_rte_rrtmgp_bw_gpu rte_rrtmgp rte_rrtmgp_cuda rte_rrtmgp_cuda_rt curand ${LIBS} m)

  add_executable(test_rt_lite_gpu test_rt_lite.cu)
  target_link_libraries(test_rt_lite_gpu rte_rrtmgp_cuda rte_rrtmgp_cuda_rt curand ${LIBS} m)
endif()

add_executable(test_rte_rrtmgp Radiation_solver.cpp test_rte_rrtmgp.cpp)
target_link_libraries(test_rte_rrtmgp rte_rrtmgp ${LIBS} m)

add_executable(test_rt_tilt_prep tilted_column.cpp test_rt_tilt_prep.cpp)
target_link_libraries(test_rt_tilt_prep rte_rrtmgp ${LIBS} m)
