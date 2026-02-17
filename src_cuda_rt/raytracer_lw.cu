#include <curand_kernel.h>

#include "raytracer_lw.h"
#include "array.h"
#include "optical_props_rt.h"

#include "raytracer_definitions.h"
#include "raytracer_functions.h"
#include "raytracer_kernels_lw.h"
#include "alias_table.h"

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
    void get_emitted_power(
            const Vector<int> grid_cells, const Vector<Float> grid_d,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa_tot,
            const Float* __restrict__ lay_source, const Float* __restrict__ sfc_source,
            const Float* __restrict__ emis_sfc, const Float tod_inc_flx,
            Float* __restrict__ power_atm,
            Float* __restrict__ power_sfc,
            Float* __restrict__ power_tod)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (ix < grid_cells.x) && (iy < grid_cells.y) && (iz < (grid_cells.z)) )
        {
            // tau = k_ext * dz; hence k_ext*cell_volume = tau * cell_area
            const int idx = ix + iy*grid_cells.x + iz*grid_cells.x*grid_cells.y;
            power_atm[idx] = Float(4.) * M_PI * tau_tot[idx] * lay_source[idx] * (Float(1.) - ssa_tot[idx]);

            if (iz == 0)
            {
                const int idx_2d = ix + iy*grid_cells.x;
                power_sfc[idx_2d] = M_PI * emis_sfc[idx_2d] * sfc_source[idx_2d];
            }
            if (iz == grid_cells.z-1)
            {
                const int idx_2d = ix + iy*grid_cells.x;
                power_tod[idx_2d] = tod_inc_flx;
            }

        }
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
            const int x1 = min(grid_cells.x-1, int(floor((grid_x+1)*fx)));
            const int y0 = grid_y*fy;
            const int y1 = min(grid_cells.y-1, int(floor((grid_y+1)*fy)));
            const int z0 = grid_z*fz;
            const int z1 = min(grid_cells.z-1, int(floor((grid_z+1)*fz)));

            const int ijk_grid = grid_x + grid_y*kn_grid.x + grid_z*kn_grid.y*kn_grid.x;
            Float k_null = k_ext_null_min;

            for (int k=z0; k<=z1; ++k)
                for (int j=y0; j<=y1; ++j)
                    for (int i=x0; i<=x1; ++i)
                    {
                        const int ijk_in = i + j*grid_cells.x + k*grid_cells.x*grid_cells.y;
                        k_null = max(k_null, k_ext[ijk_in]);
                    }

            k_null_grid[ijk_grid] = k_null;
        }
    }


    __global__
    void bundle_optical_props(
            const Vector<int> grid_cells, const Vector<Float> grid_d,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa_tot,
            const Float* __restrict__ tau_cld, const Float* __restrict__ ssa_cld, const Float* __restrict__ asy_cld,
            const Float* __restrict__ tau_aer, const Float* __restrict__ ssa_aer, const Float* __restrict__ asy_aer,
            Float* __restrict__ k_ext, Optics_scat* __restrict__ scat_asy)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol_x < grid_cells.x) && (icol_y < grid_cells.y) && (iz < (grid_cells.z)) )
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
            const Vector<int> grid_cells, const Float power_per_photon,
            const Float* __restrict__ count_1, const Float* __restrict__ count_2, const Float* __restrict__ count_3, const Float* __restrict__ count_4,
            Float* __restrict__ flux_1, Float* __restrict__ flux_2, Float* __restrict__ flux_3, Float* __restrict__ flux_4)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol_x < grid_cells.x) && (icol_y < grid_cells.y) )
        {
            const int idx = icol_x + icol_y*grid_cells.x;
            flux_1[idx] += count_1[idx] * power_per_photon;
            flux_2[idx] += count_2[idx] * power_per_photon;
            flux_3[idx] += count_3[idx] * power_per_photon;
            flux_4[idx] += count_4[idx] * power_per_photon;
        }
    }


    __global__
    void count_to_flux_3d(
            const Vector<int> grid_cells, const Vector<Float> grid_d,
            const Float power_per_photon,
            const Float* __restrict__ count,
            Float* __restrict__ flux)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( icol_x < grid_cells.x) && ( icol_y < grid_cells.y) && ( iz < grid_cells.z))
        {
            const int idx = icol_x + icol_y*grid_cells.x + iz*grid_cells.x*grid_cells.y;

            flux[idx] += count[idx] * power_per_photon / grid_d.z;
        }
    }
}

void Raytracer_lw::trace_rays(
        const int igpt,
        const bool switch_independent_column,
        const Int lw_photon_count,
        const Vector<int> grid_cells,
        const Vector<Float> grid_d,
        const Vector<int> kn_grid,
        const Array_gpu<Float,2>& tau_total,
        const Array_gpu<Float,2>& ssa_total,
        const Array_gpu<Float,2>& tau_cloud,
        const Array_gpu<Float,2>& ssa_cloud,
        const Array_gpu<Float,2>& asy_cloud,
        const Array_gpu<Float,2>& tau_aeros,
        const Array_gpu<Float,2>& ssa_aeros,
        const Array_gpu<Float,2>& asy_aeros,
        const Array_gpu<Float,2>& lay_source,
        const Array_gpu<Float,1>& sfc_source,
        const Array_gpu<Float,2>& emis_sfc,
        const Float tod_inc_flx,
        Array_gpu<Float,2>& flux_tod_dn,
        Array_gpu<Float,2>& flux_tod_up,
        Array_gpu<Float,2>& flux_sfc_dn,
        Array_gpu<Float,2>& flux_sfc_up,
        Array_gpu<Float,3>& flux_abs)
{
    const int n_lay = tau_total.dim(2);

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

    // first on the whole grid expect the extra layer
    bundle_optical_props<<<grid_3d, block_3d>>>(
            grid_cells, grid_d,
            tau_total.ptr(), ssa_total.ptr(),
            tau_cloud.ptr(), ssa_cloud.ptr(), asy_cloud.ptr(),
            tau_aeros.ptr(), ssa_aeros.ptr(), asy_aeros.ptr(),
            k_ext.ptr(), scat_asy.ptr());

    Array_gpu<Float,3> power_atm({grid_cells.x, grid_cells.y, grid_cells.z});
    Array_gpu<Float,2> power_sfc({grid_cells.x, grid_cells.y});
    Array_gpu<Float,2> power_tod({grid_cells.x, grid_cells.y});

    const int grid_z_pwr = (grid_cells.z+2)/block_z + ((grid_cells.z+2)%block_z > 0);
    dim3 grid_pwr(grid_col_x, grid_col_y, grid_z_pwr);
    dim3 block_pwr(block_col_x, block_col_y, block_z);

    get_emitted_power<<<grid_pwr, block_3d>>>(
            grid_cells, grid_d,
            tau_total.ptr(), ssa_total.ptr(), lay_source.ptr(),
            sfc_source.ptr(), emis_sfc.ptr(), tod_inc_flx,
            power_atm.ptr(), power_sfc.ptr(), power_tod.ptr());

    // Build alias tables for emission source sampling.
    const int n_atm = grid_cells.x*grid_cells.y*grid_cells.z;
    const int n_2d = grid_cells.x*grid_cells.y;

    Array_gpu<double,1> alias_prob_atm({n_atm});
    Array_gpu<int,1> alias_idx_atm({n_atm});

    Array_gpu<double,1> alias_prob_sfc({n_2d});
    Array_gpu<int,1> alias_idx_sfc({n_2d});

    Array_gpu<double,1> alias_prob_tod({n_2d});
    Array_gpu<int,1> alias_idx_tod({n_2d});

    double total_power_atm, total_power_sfc, total_power_tod;

    build_alias_table(power_atm.ptr(), n_atm, alias_prob_atm.ptr(), alias_idx_atm.ptr(), total_power_atm);
    build_alias_table(power_sfc.ptr(), n_2d,  alias_prob_sfc.ptr(), alias_idx_sfc.ptr(), total_power_sfc);
    build_alias_table(power_tod.ptr(), n_2d,  alias_prob_tod.ptr(), alias_idx_tod.ptr(), total_power_tod);

    const Float total_power = total_power_atm + total_power_sfc + total_power_tod;

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
    Array_gpu<Float,2> surface_dn_count({grid_cells.x, grid_cells.y});
    Array_gpu<Float,2> surface_up_count({grid_cells.x, grid_cells.y});
    Array_gpu<Float,3> atmos_count({grid_cells.x, grid_cells.y, grid_cells.z});

    // domain sizes
    const Vector<Float> grid_size = grid_d * grid_cells;

    // // number of photons per thread, this should a power of 2 and nonzero
    // Float photons_per_thread_tmp = std::max(Float(1), static_cast<Float>(photons_total) / (rt_kernel_grid * rt_kernel_block));
    // Int photons_per_thread = pow(Float(2.), std::floor(std::log2(photons_per_thread_tmp)));

    // // with very low number of columns and photons_per_pixel, we may have too many threads firing a single photons, actually exceeding photons_per pixel
    // // In that case, reduce grid and block size
    // Int actual_photons_per_pixel = photons_per_thread * rt_kernel_grid * rt_kernel_block / (qrng_grid_x * qrng_grid_y);

    // int rt_kernel_grid_size = rt_kernel_grid;
    // int rt_kernel_block_size = rt_kernel_block;
    // while ( (actual_photons_per_pixel > photons_per_pixel) )
    // {
    //     if (rt_kernel_grid_size > 1)
    //         rt_kernel_grid_size /= 2;
    //     else
    //         rt_kernel_block_size /= 2;

    //     photons_per_thread_tmp = std::max(Float(1), static_cast<Float>(photons_total) / (rt_kernel_grid_size * rt_kernel_block_size));
    //     photons_per_thread = pow(Float(2.), std::floor(std::log2(photons_per_thread_tmp)));
    //     actual_photons_per_pixel = photons_per_thread * rt_kernel_grid_size * rt_kernel_block_size / (qrng_grid_x * qrng_grid_y);
    // }

    Int photons_per_thread;
    Int rt_kernel_grid_size = rt_lw_kernel_grid;
    Int rt_kernel_block_size = rt_lw_kernel_block;

    if (rt_lw_kernel_grid*rt_lw_kernel_block < lw_photon_count)
    {
        photons_per_thread = lw_photon_count / (rt_lw_kernel_grid*rt_lw_kernel_block);
    }
    else if (lw_photon_count > rt_lw_kernel_block)
    {
        photons_per_thread = 1;
        rt_kernel_grid_size = lw_photon_count / rt_lw_kernel_grid;
    }
    else
    {
        photons_per_thread = 1;
        rt_kernel_grid_size = 1;
        rt_kernel_block_size = lw_photon_count;
    }

    dim3 grid(rt_kernel_grid_size);
    dim3 block(rt_kernel_block_size);

    const Float rng_offset = igpt*rt_lw_kernel_grid*rt_lw_kernel_block;

    auto run_raytracer = [&](
        const int src_type,
        const double* alias_prob,
        const int* alias_idx,
        const int n_table,
        const double total_power_src)
    {
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, tod_dn_count.ptr());
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, tod_up_count.ptr());
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, surface_dn_count.ptr());
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, surface_up_count.ptr());
        Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(grid_cells.x, grid_cells.y, grid_cells.z, atmos_count.ptr());

        ray_tracer_lw_kernel<<<grid, block>>>(
                rng_offset,
                src_type,
                switch_independent_column,
                photons_per_thread,
                alias_prob,
                alias_idx,
                n_table,
                k_null_grid.ptr(),
                tod_dn_count.ptr(),
                tod_up_count.ptr(),
                surface_dn_count.ptr(),
                surface_up_count.ptr(),
                atmos_count.ptr(),
                k_ext.ptr(), scat_asy.ptr(),
                emis_sfc.ptr(),
                grid_size, grid_d, grid_cells, kn_grid);

        const Float power_per_photon = total_power_src / (photons_per_thread * rt_lw_kernel_grid * rt_lw_kernel_block);

        count_to_flux_2d<<<grid_2d, block_2d>>>(
                grid_cells,
                power_per_photon,
                tod_dn_count.ptr(),
                tod_up_count.ptr(),
                surface_dn_count.ptr(),
                surface_up_count.ptr(),
                flux_tod_dn.ptr(),
                flux_tod_up.ptr(),
                flux_sfc_dn.ptr(),
                flux_sfc_up.ptr());

        count_to_flux_3d<<<grid_3d, block_3d>>>(
                grid_cells, grid_d,
                power_per_photon,
                atmos_count.ptr(),
                flux_abs.ptr());
    };

    run_raytracer(0, alias_prob_atm.ptr(), alias_idx_atm.ptr(), n_atm, total_power_atm);
    run_raytracer(1, alias_prob_sfc.ptr(), alias_idx_sfc.ptr(), n_2d,  total_power_sfc);
    run_raytracer(2, alias_prob_tod.ptr(), alias_idx_tod.ptr(), n_2d,  total_power_tod);
}

Raytracer_lw::Raytracer_lw()
{
//    curandDirectionVectors32_t* qrng_vectors;
//    curandGetDirectionVectors32(
//                &qrng_vectors,
//                CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
//    unsigned int* qrng_constants;
//    curandGetScrambleConstants32(&qrng_constants);
//
//    this->qrng_vectors_gpu = allocate_gpu<curandDirectionVectors32_t>(2);
//    this->qrng_constants_gpu = allocate_gpu<unsigned int>(2);
//
//    copy_to_gpu(qrng_vectors_gpu, qrng_vectors, 2);
//    copy_to_gpu(qrng_constants_gpu, qrng_constants, 2);
}
