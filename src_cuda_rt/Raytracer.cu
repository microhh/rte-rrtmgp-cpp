#include <curand_kernel.h>

#include "Raytracer.h"
#include "Array.h"
#include "Optical_props_rt.h"

#include "raytracer_definitions.h"
#include "raytracer_functions.h"
#include "raytracer_kernels.h"

#include "gas_optics_rrtmgp_kernels_cuda_rt.h"


namespace
{
    using namespace Raytracer_functions;

    template<typename T>
    T* allocate_gpu(const int length)
    {
        T* data_ptr = Tools_gpu::allocate_gpu<T>(length);
        return data_ptr;
    }


    template<typename T>
    void copy_to_gpu(T* gpu_data, const T* cpu_data, const int length)
    {
        cuda_safe_call(cudaMemcpy(gpu_data, cpu_data, length*sizeof(T), cudaMemcpyHostToDevice));
    }


    template<typename T>
    void copy_from_gpu(T* cpu_data, const T* gpu_data, const int length)
    {
        cuda_safe_call(cudaMemcpy(cpu_data, gpu_data, length*sizeof(T), cudaMemcpyDeviceToHost));
    }


    __global__
    void create_knull_grid(
            const Vector<int> grid_cells, const Float k_ext_null_min,
            const Vector<int> kn_grid,
            const Float* __restrict__ k_ext, Float* __restrict__ k_null_grid)
    {
        const int grid_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int grid_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int grid_z = blockIdx.z*blockDim.z + threadIdx.z;
        if ( ( grid_x < kn_grid.x) && ( grid_y < kn_grid.y) && ( grid_z < kn_grid.z))
        {
            const Float fx = Float(grid_cells.x) / Float(kn_grid.x);
            const Float fy = Float(grid_cells.y) / Float(kn_grid.y);
            const Float fz = Float(grid_cells.z) / Float(kn_grid.z);

            const int x0 = grid_x*fx;
            const int x1 = floor((grid_x+1)*fx);
            const int y0 = grid_y*fy;
            const int y1 = floor((grid_y+1)*fy);
            const int z0 = grid_z*fz;
            const int z1 = floor((grid_z+1)*fz);

            const int ijk_grid = grid_x + grid_y*kn_grid.x + grid_z*kn_grid.y*kn_grid.x;
            Float k_null = k_ext_null_min;

            for (int k=z0; k<z1; ++k)
                for (int j=y0; j<y1; ++j)
                    for (int i=x0; i<x1; ++i)
                    {
                        const int ijk_in = i + j*grid_cells.x + k*grid_cells.x*grid_cells.y;
                        k_null = max(k_null, k_ext[ijk_in]);
                    }

            k_null_grid[ijk_grid] = k_null;
        }
    }


    __global__
    void bundles_optical_props(
            const Vector<int> grid_cells, const Vector<Float> grid_d,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa_tot,
            const Float* __restrict__ tau_cld, const Float* __restrict__ ssa_cld, const Float* __restrict__ asy_cld,
            const Float* __restrict__ tau_aer, const Float* __restrict__ ssa_aer, const Float* __restrict__ asy_aer,
            Float* __restrict__ k_ext, Optics_scat* __restrict__ scat_asy)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol_x < grid_cells.x) && (icol_y < grid_cells.y) && (iz < grid_cells.z) )
        {
            const int idx = icol_x + icol_y*grid_cells.x + iz*grid_cells.y*grid_cells.x;
            const Float kext_tot = tau_tot[idx] / grid_d.z;
            const Float kext_cld = tau_cld[idx] / grid_d.z;
            const Float kext_aer = tau_aer[idx] / grid_d.z;
            const Float ksca_cld = kext_cld * ssa_cld[idx];
            const Float ksca_aer = kext_aer * ssa_aer[idx];
            const Float ksca_gas = kext_tot * ssa_tot[idx] - ksca_cld - ksca_aer;
            k_ext[idx] = tau_tot[idx] / grid_d.z;

            scat_asy[idx].k_sca_gas = ksca_gas;
            scat_asy[idx].k_sca_cld = ksca_cld;
            scat_asy[idx].k_sca_aer = ksca_aer;
            scat_asy[idx].asy_cld = asy_cld[idx];
            scat_asy[idx].asy_aer = asy_aer[idx];
        }
    }


    __global__
    void count_to_flux_2d(
            const Vector<int> grid_cells, const Float photons_per_col, const Float toa_src,
            const Float* __restrict__ count_1, const Float* __restrict__ count_2, const Float* __restrict__ count_3, const Float* __restrict__ count_4, const Float* __restrict__ count_5,
            Float* __restrict__ flux_1, Float* __restrict__ flux_2, Float* __restrict__ flux_3, Float* __restrict__ flux_4, Float* __restrict__ flux_5)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( icol_x < grid_cells.x) && ( icol_y < grid_cells.y) )
        {
            const int idx = icol_x + icol_y*grid_cells.x;
            const Float flux_per_ray = toa_src / photons_per_col;
            flux_1[idx] = count_1[idx] * flux_per_ray;
            flux_2[idx] = count_2[idx] * flux_per_ray;
            flux_3[idx] = count_3[idx] * flux_per_ray;
            flux_4[idx] = count_4[idx] * flux_per_ray;
            flux_5[idx] = count_5[idx] * flux_per_ray;
        }
    }

    __global__
    void count_to_flux_3d(
            const Vector<int> grid_cells, const Float photons_per_col,
            const Vector<Float> grid_d, const Float toa_src,
            const Float* __restrict__ count_1, const Float* __restrict__ count_2,
            Float* __restrict__ flux_1, Float* __restrict__ flux_2)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( icol_x < grid_cells.x) && ( icol_y < grid_cells.y) && ( iz < grid_cells.z))
        {
            const int idx = icol_x + icol_y*grid_cells.x + iz*grid_cells.x*grid_cells.y;
            const Float flux_per_ray = toa_src / photons_per_col;
            flux_1[idx] = count_1[idx] * flux_per_ray / grid_d.z;
            flux_2[idx] = count_2[idx] * flux_per_ray / grid_d.z;
        }
    }
}


Raytracer::Raytracer()
{
    curandDirectionVectors32_t* qrng_vectors;
    curandGetDirectionVectors32(
                &qrng_vectors,
                CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    unsigned int* qrng_constants;
    curandGetScrambleConstants32(&qrng_constants);

    this->qrng_vectors_gpu = allocate_gpu<curandDirectionVectors32_t>(2);
    this->qrng_constants_gpu = allocate_gpu<unsigned int>(2);

    copy_to_gpu(qrng_vectors_gpu, qrng_vectors, 2);
    copy_to_gpu(qrng_constants_gpu, qrng_constants, 2);
}


void Raytracer::trace_rays(
        const int igpt,
        const Int photons_per_pixel,
        const Vector<int> grid_cells,
        const Vector<Float> grid_d,
        const Vector<int> kn_grid,
        const Array_gpu<Float,2>& mie_cdf,
        const Array_gpu<Float,3>& mie_ang,
        const Array_gpu<Float,2>& tau_total,
        const Array_gpu<Float,2>& ssa_total,
        const Array_gpu<Float,2>& tau_cloud,
        const Array_gpu<Float,2>& ssa_cloud,
        const Array_gpu<Float,2>& asy_cloud,
        const Array_gpu<Float,2>& tau_aeros,
        const Array_gpu<Float,2>& ssa_aeros,
        const Array_gpu<Float,2>& asy_aeros,
        const Array_gpu<Float,2>& r_eff,
        const Array_gpu<Float,2>& surface_albedo,
        const Float zenith_angle,
        const Float azimuth_angle,
        const Float tod_inc_direct,
        const Float tod_inc_diffuse,
        Array_gpu<Float,2>& flux_tod_dn,
        Array_gpu<Float,2>& flux_tod_up,
        Array_gpu<Float,2>& flux_sfc_dir,
        Array_gpu<Float,2>& flux_sfc_dif,
        Array_gpu<Float,2>& flux_sfc_up,
        Array_gpu<Float,3>& flux_abs_dir,
        Array_gpu<Float,3>& flux_abs_dif)
{
    // set of block and grid dimensions used in data processing kernels - requires some proper tuning later
    const int block_col_x = 8;
    const int block_col_y = 8;
    const int block_z = 4;

    const int grid_col_x = grid_cells.x/block_col_x + (grid_cells.x%block_col_x > 0);
    const int grid_col_y = grid_cells.y/block_col_y + (grid_cells.y%block_col_y > 0);
    const int grid_z = grid_cells.z/block_z + (grid_cells.z%block_z > 0);

    dim3 grid_2d(grid_col_x, grid_col_y);
    dim3 block_2d(block_col_x, block_col_y);
    dim3 grid_3d(grid_col_x, grid_col_y, grid_z);
    dim3 block_3d(block_col_x, block_col_y, block_z);

    // Bundle optical properties in struct
    Array_gpu<Float,3> k_ext({grid_cells.x, grid_cells.y, grid_cells.z});
    Array_gpu<Optics_scat,3> scat_asy({grid_cells.x, grid_cells.y, grid_cells.z});

    bundles_optical_props<<<grid_3d, block_3d>>>(
            grid_cells, grid_d,
            tau_total.ptr(), ssa_total.ptr(),
            tau_cloud.ptr(), ssa_cloud.ptr(), asy_cloud.ptr(),
            tau_aeros.ptr(), ssa_aeros.ptr(), asy_aeros.ptr(),
            k_ext.ptr(), scat_asy.ptr());

    // create k_null_grid
    const int block_kn_x = 8;
    const int block_kn_y = 8;
    const int block_kn_z = 4;

    const int grid_kn_x = kn_grid.x/block_kn_x + (kn_grid.x%block_kn_x > 0);
    const int grid_kn_y = kn_grid.y/block_kn_y + (kn_grid.y%block_kn_y > 0);
    const int grid_kn_z = kn_grid.z/block_kn_z + (kn_grid.z%block_kn_z > 0);

    dim3 grid_kn(grid_kn_x, grid_kn_y, grid_kn_z);
    dim3 block_kn(block_kn_x, block_kn_y, block_kn_z);

    Array_gpu<Float,3> k_null_grid({kn_grid.x, kn_grid.y, kn_grid.z});
    const Float k_ext_null_min = Float(1e-3);

    create_knull_grid<<<grid_kn, block_kn>>>(
            grid_cells, k_ext_null_min,
            kn_grid,
            k_ext.ptr(), k_null_grid.ptr());

    // initialise output arrays and set to 0
    Array_gpu<Float,2> tod_dn_count({grid_cells.x, grid_cells.y});
    Array_gpu<Float,2> tod_up_count({grid_cells.x, grid_cells.y});
    Array_gpu<Float,2> surface_down_direct_count({grid_cells.x, grid_cells.y});
    Array_gpu<Float,2> surface_down_diffuse_count({grid_cells.x, grid_cells.y});
    Array_gpu<Float,2> surface_up_count({grid_cells.x, grid_cells.y});
    Array_gpu<Float,3> atmos_direct_count({grid_cells.x, grid_cells.y, grid_cells.z});
    Array_gpu<Float,3> atmos_diffuse_count({grid_cells.x, grid_cells.y, grid_cells.z});

    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, tod_dn_count.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, tod_up_count.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, surface_down_direct_count.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, surface_down_diffuse_count.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, surface_up_count.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, grid_cells.z, atmos_direct_count.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, grid_cells.z, atmos_diffuse_count.ptr());

    // domain sizes
    const Vector<Float> grid_size = grid_d * grid_cells;

    // direction of direct rays. Take into account that azimuth is 0 north and increases clockwise
    const Vector<Float> sun_direction = {-std::sin(zenith_angle) * std::cos(Float(0.5*M_PI) - azimuth_angle),
                                         -std::sin(zenith_angle) * std::sin(Float(0.5*M_PI) - azimuth_angle),
                                         -std::cos(zenith_angle)};

    // smallest two power that is larger than grid dimension (minimum of 2 is currently required)
    const Int qrng_grid_x = std::max(Float(2), pow(Float(2.), ceil(std::log2(Float(grid_cells.x)))) );
    const Int qrng_grid_y = std::max(Float(2), pow(Float(2.), ceil(std::log2(Float(grid_cells.y)))) );
   
    // total number of photons
    const Int photons_total = photons_per_pixel * qrng_grid_x * qrng_grid_y;

    // number of photons per thread, this should a power of 2 and nonzero
    Float photons_per_thread_tmp = std::max(Float(1), static_cast<Float>(photons_total) / (rt_kernel_grid * rt_kernel_block));
    Int photons_per_thread = pow(Float(2.), std::floor(std::log2(photons_per_thread_tmp)));
    
    dim3 grid,block;
    
    // with very low number of columns and photons_per_pixel, we may have too many threads firing a single photons, actually exceeding photons_per pixel
    // In that case, reduce grid and block size
    Int actual_photons_per_pixel = photons_per_thread * rt_kernel_grid * rt_kernel_block / (qrng_grid_x * qrng_grid_y);
    
    int rt_kernel_grid_size = rt_kernel_grid;
    int rt_kernel_block_size = rt_kernel_block;
    while ( (actual_photons_per_pixel > photons_per_pixel) )
    {
        if (rt_kernel_grid_size > 1)
            rt_kernel_grid_size /= 2;
        else
            rt_kernel_block_size /= 2;        
        
        photons_per_thread_tmp = std::max(Float(1), static_cast<Float>(photons_total) / (rt_kernel_grid_size * rt_kernel_block_size));
        photons_per_thread = pow(Float(2.), std::floor(std::log2(photons_per_thread_tmp)));
        actual_photons_per_pixel = photons_per_thread * rt_kernel_grid_size * rt_kernel_block_size / (qrng_grid_x * qrng_grid_y);
    }
    
    grid = {rt_kernel_grid_size};
    block = {rt_kernel_block_size};
    
    const int mie_table_size = mie_cdf.size();
    
    const int qrng_gpt_offset = (igpt-1) * rt_kernel_grid_size * rt_kernel_block_size * photons_per_thread;
    ray_tracer_kernel<<<grid, block,sizeof(Float)*mie_table_size>>>(
            photons_per_thread,
            qrng_grid_x,
            qrng_grid_y,
            qrng_gpt_offset,
            k_null_grid.ptr(),
            tod_dn_count.ptr(),
            tod_up_count.ptr(),
            surface_down_direct_count.ptr(),
            surface_down_diffuse_count.ptr(),
            surface_up_count.ptr(),
            atmos_direct_count.ptr(),
            atmos_diffuse_count.ptr(),
            k_ext.ptr(), scat_asy.ptr(),
            r_eff.ptr(),
            tod_inc_direct,
            tod_inc_diffuse,
            surface_albedo.ptr(),
            grid_size, grid_d, grid_cells, kn_grid,
            sun_direction,
            this->qrng_vectors_gpu, this->qrng_constants_gpu,
            mie_cdf.ptr(), mie_ang.ptr(), mie_table_size);

    // convert counts to fluxes

    const Float toa_src = tod_inc_direct + tod_inc_diffuse;
    count_to_flux_2d<<<grid_2d, block_2d>>>(
            grid_cells, photons_per_pixel,
            toa_src,
            tod_dn_count.ptr(),
            tod_up_count.ptr(),
            surface_down_direct_count.ptr(),
            surface_down_diffuse_count.ptr(),
            surface_up_count.ptr(),
            flux_tod_dn.ptr(),
            flux_tod_up.ptr(),
            flux_sfc_dir.ptr(),
            flux_sfc_dif.ptr(),
            flux_sfc_up.ptr());

    count_to_flux_3d<<<grid_3d, block_3d>>>(
            grid_cells, photons_per_pixel,
            grid_d,
            toa_src,
            atmos_direct_count.ptr(),
            atmos_diffuse_count.ptr(),
            flux_abs_dir.ptr(),
            flux_abs_dif.ptr());





}
