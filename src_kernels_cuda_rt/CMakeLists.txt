FILE(GLOB sourcefiles_rt
    "../src_kernels_cuda_rt/gas_optics_rrtmgp_kernels_launchers_rt.cu"
    "../src_kernels_cuda_rt/optical_props_kernels_launchers_rt.cu"
    "../src_kernels_cuda_rt/rte_solver_kernels_launchers_rt.cu"
    "../src_kernels_cuda_rt/subset_kernels_launchers_rt.cu"
    "../src_kernels_cuda_rt/gpt_combine_kernels_launchers_rt.cu"
    "../src_kernels_cuda_rt/raytracer_kernels_sw.cu"
    "../src_kernels_cuda_rt/raytracer_kernels_lw.cu"
    "../src_kernels_cuda_rt/raytracer_kernels_bw.cu")


if(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    set_source_files_properties("../src_kernels_cuda_rt/raytracer_kernels_sw.cu" PROPERTIES COMPILE_FLAGS --use_fast_math)
    set_source_files_properties("../src_kernels_cuda_rt/raytracer_kernels_lw.cu" PROPERTIES COMPILE_FLAGS --use_fast_math)
    set_source_files_properties("../src_kernels_cuda_rt/raytracer_kernels_bw.cu" PROPERTIES COMPILE_FLAGS --use_fast_math)
endif()

include_directories("../include" "../include_rt" "../include_rt_kernels" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_kernels_cuda_rt STATIC ${sourcefiles_rt})

# These explicitly instantiated kernel templates are launched from src_cuda_rt.
# CUDA 13 defaults to static host stubs, which cannot link across those files.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set_property(SOURCE
        "../src_kernels_cuda_rt/raytracer_kernels_sw.cu"
        "../src_kernels_cuda_rt/raytracer_kernels_lw.cu"
        "../src_kernels_cuda_rt/raytracer_kernels_bw.cu"
        APPEND PROPERTY COMPILE_OPTIONS --static-global-template-stub=false)
endif()

