#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
include_directories(${INCLUDE_DIRS} "../include" "../include_test")

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
  cuda_add_executable(test_rte_rrtmgp Radiation_solver.cpp test_rte_rrtmgp.cpp)
  target_link_libraries(test_rte_rrtmgp rte_rrtmgp ${LIBS} m)
else()
  add_executable(test_rte_rrtmgp Radiation_solver.cpp test_rte_rrtmgp.cpp)
  target_link_libraries(test_rte_rrtmgp rte_rrtmgp ${LIBS} m)
endif()
