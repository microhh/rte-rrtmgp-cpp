#include <curand_kernel.h>

#include "Optical_props_rt.h"
#include "Array.h"

#include "Raytracer_bw.h"
#include "raytracer_kernels_bw.h"
#include "gas_optics_rrtmgp_kernels_cuda_rt.h"


namespace
{
    __global__
    void normalize_xyz_camera_kernel(
            const Camera camera,
            const Float total_source,
            Float* __restrict__ XYZ)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        if ( ( ix < camera.nx) && ( iy < camera.ny) )
        {
            for (int i=0; i<3; ++i)
            {
                const int idx_out = ix + iy*camera.nx + i*camera.nx*camera.ny;
                XYZ[idx_out] /= total_source;
            }
        }
    }

    __global__
    void add_camera_kernel(
            const Camera camera,
            const Float* __restrict__ flux_camera,
            Float* __restrict__ radiance)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        if ( ( ix < camera.nx) && ( iy < camera.ny) )
        {
            const int idx = ix + iy*camera.nx;
            radiance[idx] += flux_camera[idx];
        }
    }

    __global__
    void add_xyz_camera_kernel(
            const Camera camera,
            const Float* __restrict__ xyz_factor,
            const Float* __restrict__ flux_camera,
            Float* __restrict__ XYZ)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        if ( ( ix < camera.nx) && ( iy < camera.ny) )
        {
            const int idx_in = ix + iy*camera.nx;
            for (int i=0; i<3; ++i)
            {
                const int idx_out = ix + iy*camera.nx + i*camera.nx*camera.ny;
                XYZ[idx_out] += xyz_factor[i] * flux_camera[idx_in];
            }
        }
    }

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
            const Vector<int> grid_cells, const Vector<int> kn_grid, const Float k_ext_null_min,
            const Float* __restrict__ k_ext, Grid_knull* __restrict__ k_null_grid)
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
            Float k_null_min = Float(1e15); // just a ridicilously high value
            Float k_null_max = Float(0.);
           
            for (int k=z0; k<=z1; ++k)
                for (int j=y0; j<=y1; ++j)
                    for (int i=x0; i<=x1; ++i)
                    {
                        const int ijk_in = i + j*grid_cells.x + k*grid_cells.x*grid_cells.y;
                        k_null_min = min(k_null_min, k_ext[ijk_in]);
                        k_null_max = max(k_null_max, k_ext[ijk_in]);
                    }

            if (k_null_min == k_null_max) k_null_min = k_null_max * Float(0.99);
            k_null_grid[ijk_grid].k_min = k_null_min;
            k_null_grid[ijk_grid].k_max = k_null_max;
        }
    }


    __global__
    void bundles_optical_props(
            const Vector<int> grid_cells, const int nlay, const Float grid_dz,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa_tot,
            const Float* __restrict__ tau_cld, const Float* __restrict__ ssa_cld, const Float* __restrict__ asy_cld,
            const Float* __restrict__ tau_aer, const Float* __restrict__ ssa_aer, const Float* __restrict__ asy_aer,
            const Float rayleigh,
            const Float* __restrict__ col_dry, const Float* __restrict__ vmr_h2o,
            Float* __restrict__ k_ext, Optics_scat* __restrict__ scat_asy)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz     = blockIdx.z*blockDim.z + threadIdx.z;
        if ( (icol_x < grid_cells.x) && (icol_y < grid_cells.y) && (iz < grid_cells.z) )
        {
            const int idx = icol_x + icol_y*grid_cells.x + iz*grid_cells.y*grid_cells.x;

            const Float kext_cld = tau_cld[idx] / grid_dz;
            const Float kext_aer = tau_aer[idx] / grid_dz;
            const Float ksca_cld = kext_cld * ssa_cld[idx];
            const Float ksca_aer = kext_aer * ssa_aer[idx];
            const Float ksca_gas = (rayleigh > 0) ? rayleigh * (1 + vmr_h2o[idx]) * col_dry[idx] / grid_dz :
                                                    tau_tot[idx] / grid_dz * ssa_tot[idx] - ksca_cld - ksca_aer;

            const Float kext_tot_old = tau_tot[idx] / grid_dz;
            const Float kext_gas_old = kext_tot_old - kext_cld - kext_aer;
            const Float kabs_gas = kext_gas_old - (kext_tot_old * ssa_tot[idx] - ksca_cld - ksca_aer);
            const Float kext_gas = kabs_gas + ksca_gas;

            k_ext[idx] = kext_cld + kext_gas + kext_aer;
            scat_asy[idx].k_sca_gas = ksca_gas;
            scat_asy[idx].k_sca_cld = ksca_cld;
            scat_asy[idx].k_sca_aer = ksca_aer;
            scat_asy[idx].asy_cld = asy_cld[idx];
            scat_asy[idx].asy_aer = asy_aer[idx];
        }

    }

    __global__
    void bundles_optical_props_bb(
            const Vector<int> grid_cells, const int nlay, const Float grid_dz,
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
            const Float kext_cld = tau_cld[idx] / grid_dz;
            const Float kext_aer = tau_aer[idx] / grid_dz;
            const Float kext_tot = tau_tot[idx] / grid_dz;
            const Float ksca_cld = kext_cld * ssa_cld[idx];
            const Float ksca_aer = kext_aer * ssa_aer[idx];

            k_ext[idx] = tau_tot[idx] / grid_dz;
            scat_asy[idx].k_sca_gas = kext_tot*ssa_tot[idx] - ksca_cld - ksca_aer;
            scat_asy[idx].k_sca_cld = ksca_cld;
            scat_asy[idx].k_sca_aer = ksca_aer;
            scat_asy[idx].asy_cld = asy_cld[idx];
            scat_asy[idx].asy_aer = asy_aer[idx];
        }
    }

    __global__
    void background_profile(
            const Vector<int> grid_cells, const int nbg,
            const Float* __restrict__ z_lev,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa_tot,
            const Float* __restrict__ tau_cld, const Float* __restrict__ ssa_cld, const Float* __restrict__ asy_cld,
            const Float* __restrict__ tau_aer, const Float* __restrict__ ssa_aer, const Float* __restrict__ asy_aer,
            const Float rayleigh,
            const Float* __restrict__ col_dry, const Float* __restrict__ vmr_h2o,
            Float* __restrict__ k_ext_bg, Optics_scat* __restrict__ scat_asy_bg, Float* __restrict__ z_lev_bg)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < nbg)
        {
            const int idx = (i+grid_cells.z)*grid_cells.y*grid_cells.x;
            const Float dz = abs(z_lev[i+grid_cells.z+1] - z_lev[i+grid_cells.z]);

            const Float kext_cld = tau_cld[idx] / dz;
            const Float kext_aer = tau_aer[idx] / dz;
            const Float ksca_cld = kext_cld * ssa_cld[idx];
            const Float ksca_aer = kext_aer * ssa_aer[idx];
            const Float ksca_gas = (rayleigh > 0) ? rayleigh * (1 + vmr_h2o[idx]) * col_dry[idx] / dz :
                                                    tau_tot[idx] / dz * ssa_tot[idx] - ksca_cld - ksca_aer;

            const Float kext_tot_old = tau_tot[idx] / dz;
            const Float kext_gas_old = kext_tot_old - kext_cld  - kext_aer;
            const Float kabs_gas = kext_gas_old - (kext_tot_old * ssa_tot[idx] - ksca_cld - ksca_aer);
            const Float kext_gas = kabs_gas + ksca_gas;

            k_ext_bg[i] = kext_cld + kext_gas + kext_aer;
            scat_asy_bg[i].k_sca_gas = ksca_gas;
            scat_asy_bg[i].k_sca_cld = ksca_cld;
            scat_asy_bg[i].k_sca_aer = ksca_aer;
            scat_asy_bg[i].asy_cld = asy_cld[idx];
            scat_asy_bg[i].asy_aer = asy_aer[idx];

            z_lev_bg[i] = z_lev[i + grid_cells.z];
            if (i == nbg-1) z_lev_bg[i + 1] = z_lev[i + grid_cells.z + 1];
        }
    }

    __global__
    void background_profile_bb(
            const Vector<int> grid_cells, const int nbg,
            const Float* __restrict__ z_lev,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa_tot,
            const Float* __restrict__ tau_cld, const Float* __restrict__ ssa_cld, const Float* __restrict__ asy_cld,
            const Float* __restrict__ tau_aer, const Float* __restrict__ ssa_aer, const Float* __restrict__ asy_aer,
            Float* __restrict__ k_ext_bg, Optics_scat* __restrict__ scat_asy_bg, Float* __restrict__ z_lev_bg)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < nbg)
        {
            const int idx = (i+grid_cells.z)*grid_cells.y*grid_cells.x;
            const Float dz = abs(z_lev[i+grid_cells.z+1] - z_lev[i+grid_cells.z]);

            const Float kext_cld = tau_cld[idx] / dz;
            const Float kext_aer = tau_aer[idx] / dz;
            const Float kext_tot = tau_tot[idx] / dz;

            const Float ksca_cld = kext_cld * ssa_cld[idx];
            const Float ksca_aer = kext_aer * ssa_aer[idx];
            const Float ksca_gas = kext_tot * ssa_tot[idx] - ksca_cld - ksca_aer;

            k_ext_bg[i] = kext_tot;
            scat_asy_bg[i].k_sca_gas = ksca_gas;
            scat_asy_bg[i].k_sca_cld = ksca_cld;
            scat_asy_bg[i].k_sca_aer = ksca_aer;
            scat_asy_bg[i].asy_cld = asy_cld[idx];
            scat_asy_bg[i].asy_aer = asy_aer[idx];

            z_lev_bg[i] = z_lev[i + grid_cells.z];
            if (i == nbg-1) z_lev_bg[i + 1] = z_lev[i + grid_cells.z + 1];
        }
    }

    __global__
    void count_to_flux_2d(
            const Camera camera, const Float photons_per_pixel, const Float toa_src, const Float toa_factor,
            const Float* __restrict__ count, Float* __restrict__ flux)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( ix < camera.nx) && ( iy < camera.ny) )
        {
            const int idx = ix + iy*camera.nx;
            const Float flux_per_ray = toa_src * toa_factor / photons_per_pixel;
            flux[idx] = count[idx] * flux_per_ray;
        }
    }
}


Raytracer_bw::Raytracer_bw()
{
}

void Raytracer_bw::add_camera(
    const Camera& camera,
    const Array_gpu<Float,2>& flux_camera,
    Array_gpu<Float,2>& radiance)
{
    const int block_x = 8;
    const int block_y = 8;

    const int grid_x  = camera.nx/block_x + (camera.nx%block_x > 0);
    const int grid_y  = camera.ny/block_y + (camera.ny%block_y > 0);

    dim3 grid(grid_x, grid_y);
    dim3 block(block_x, block_y);

    add_camera_kernel<<<grid, block>>>(
            camera,
            flux_camera.ptr(),
            radiance.ptr());
}

void Raytracer_bw::add_xyz_camera(
    const Camera& camera,
    const Array_gpu<Float,1>& xyz_factor,
    const Array_gpu<Float,2>& flux_camera,
    Array_gpu<Float,3>& XYZ)
{
    const int block_x = 8;
    const int block_y = 8;

    const int grid_x  = camera.nx/block_x + (camera.nx%block_x > 0);
    const int grid_y  = camera.ny/block_y + (camera.ny%block_y > 0);

    dim3 grid(grid_x, grid_y);
    dim3 block(block_x, block_y);

    add_xyz_camera_kernel<<<grid, block>>>(
            camera,
            xyz_factor.ptr(),
            flux_camera.ptr(),
            XYZ.ptr());
}


void Raytracer_bw::normalize_xyz_camera(
    const Camera& camera,
    const Float total_source,
    Array_gpu<Float,3>& XYZ)
{
    const int block_x = 8;
    const int block_y = 8;

    const int grid_x  = camera.nx/block_x + (camera.nx%block_x > 0);
    const int grid_y  = camera.ny/block_y + (camera.ny%block_y > 0);

    dim3 grid(grid_x, grid_y);
    dim3 block(block_x, block_y);

    normalize_xyz_camera_kernel<<<grid, block>>>(
            camera,
            total_source,
            XYZ.ptr());

}


void Raytracer_bw::trace_rays(
        const int igpt,
        const int photons_per_pixel,
        const int nlay,
        const Vector<int>& grid_cells,
        const Vector<Float>& grid_d,
        const Vector<int>& kn_grid,
        const Array_gpu<Float,1>& z_lev,
        const Array_gpu<Float,3>& mie_cdf,
        const Array_gpu<Float,4>& mie_ang,
        const Array_gpu<Float,4>& mie_phase,
        const Array_gpu<Float,1>& mie_phase_ang,
        const Array_gpu<Float,2>& r_eff,
        const Array_gpu<Float,2>& tau_total,
        const Array_gpu<Float,2>& ssa_total,
        const Array_gpu<Float,2>& tau_cloud,
        const Array_gpu<Float,2>& ssa_cloud,
        const Array_gpu<Float,2>& asy_cloud,
        const Array_gpu<Float,2>& tau_aeros,
        const Array_gpu<Float,2>& ssa_aeros,
        const Array_gpu<Float,2>& asy_aeros,
        const Array_gpu<Float,2>& surface_albedo,
        const Array_gpu<Float,1>& land_use_map,
        const Float zenith_angle,
        const Float azimuth_angle,
        const Float toa_src,
        const Float toa_factor,
        const Float rayleigh,
        const Array_gpu<Float,2>& col_dry,
        const Array_gpu<Float,2>& vmr_h2o,
        const Camera& camera,
        Array_gpu<Float,2>& flux_camera)
{
    const Float mu = std::abs(std::cos(zenith_angle));

    // set of block and grid dimensions used in data processing kernels - requires some proper tuning later
    const int block_col_x = 8;
    const int block_col_y = 8;
    const int block_z = 4;

    const int grid_col_x  = grid_cells.x/block_col_x + (grid_cells.x%block_col_x > 0);
    const int grid_col_y  = grid_cells.y/block_col_y + (grid_cells.y%block_col_y > 0);
    const int grid_z  = grid_cells.z/block_z + (grid_cells.z%block_z > 0);

    dim3 grid_3d(grid_col_x, grid_col_y, grid_z);
    dim3 block_3d(block_col_x, block_col_y, block_z);

    // bundle optical properties in struct
    Array_gpu<Float,3> k_ext({grid_cells.x, grid_cells.y, grid_cells.z});
    Array_gpu<Optics_scat,3> ssa_asy({grid_cells.x, grid_cells.y, grid_cells.z});

    bundles_optical_props<<<grid_3d, block_3d>>>(
            grid_cells, nlay, grid_d.z,
            tau_total.ptr(), ssa_total.ptr(),
            tau_cloud.ptr(), ssa_cloud.ptr(), asy_cloud.ptr(),
            tau_aeros.ptr(), ssa_aeros.ptr(), asy_aeros.ptr(),
            rayleigh, col_dry.ptr(), vmr_h2o.ptr(), k_ext.ptr(), ssa_asy.ptr());

    // create k_null_grid
    const int block_kn_x = 2;
    const int block_kn_y = 2;
    const int block_kn_z = 4;

    const int grid_kn_x  = kn_grid.x/block_kn_x + (kn_grid.x%block_kn_x > 0);
    const int grid_kn_y  = kn_grid.y/block_kn_y + (kn_grid.y%block_kn_y > 0);
    const int grid_kn_z  = kn_grid.z/block_kn_z + (kn_grid.z%block_kn_z > 0);

    dim3 grid_kn(grid_kn_x, grid_kn_y, grid_kn_z);
    dim3 block_kn(block_kn_x, block_kn_y, block_kn_z);

    Array_gpu<Grid_knull,3> k_null_grid({kn_grid.x, kn_grid.y, kn_grid.z});
    const Float k_ext_null_min = Float(1e-3);

    create_knull_grid<<<grid_kn, block_kn>>>(
            grid_cells, kn_grid, k_ext_null_min,
            k_ext.ptr(), k_null_grid.ptr());

    // TOA-TOD profile (at x=0, y=0)
    const int nbg = nlay-grid_cells.z;
    Array_gpu<Float,1> k_ext_bg({nbg});
    Array_gpu<Optics_scat,1> ssa_asy_bg({nbg});
    Array_gpu<Float,1> z_lev_bg({nbg+1});

    const int block_1d_z = 16;
    const int grid_1d_z  = nbg/block_1d_z + (nbg%block_1d_z > 0);
    dim3 grid_1d(grid_1d_z);
    dim3 block_1d(block_1d_z);

    background_profile<<<grid_1d, block_1d>>>(
            grid_cells, nbg, z_lev.ptr(),
            tau_total.ptr(), ssa_total.ptr(),
            tau_cloud.ptr(), ssa_cloud.ptr(), asy_cloud.ptr(),
            tau_aeros.ptr(), ssa_aeros.ptr(), asy_aeros.ptr(),
            rayleigh, col_dry.ptr(), vmr_h2o.ptr(),
            k_ext_bg.ptr(), ssa_asy_bg.ptr(), z_lev_bg.ptr());

    Array_gpu<Float,2> camera_count({camera.nx, camera.ny});
    Array_gpu<Float,2> shot_count({camera.nx, camera.ny});
    Array_gpu<int,1> counter({1});

    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(camera.nx, camera.ny, camera_count.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(camera.nx, camera.ny, shot_count.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(1, counter.ptr());

    // domain sizes
    const Vector<Float> grid_size = grid_d * grid_cells;

    // direction of direct sun rays
    const Vector<Float> sun_direction = {std::sin(zenith_angle) * std::sin(azimuth_angle),
                                         std::sin(zenith_angle) * std::cos(azimuth_angle),
                                         std::cos(zenith_angle)};

    dim3 grid{bw_kernel_grid}, block{bw_kernel_block};

    const int mie_table_size = mie_cdf.size();

    ray_tracer_kernel_bw<<<grid, block, nbg*sizeof(Float) + 2 * sizeof(Float)*mie_table_size>>>(
            igpt-1,
            photons_per_pixel, k_null_grid.ptr(),
            camera_count.ptr(),
            shot_count.ptr(),
            counter.ptr(),
            k_ext.ptr(), ssa_asy.ptr(),
            k_ext_bg.ptr(), ssa_asy_bg.ptr(),
            z_lev_bg.ptr(),
            r_eff.ptr(),
            surface_albedo.ptr(),
            land_use_map.ptr(),
            mu,
            grid_size, grid_d, grid_cells, kn_grid,
            sun_direction, camera, nbg,
            mie_cdf.ptr(), mie_ang.ptr(),
            mie_phase.ptr(), mie_phase_ang.ptr(), mie_table_size);

    //// convert counts to fluxes
    const int block_cam_x = 8;
    const int block_cam_y = 8;

    const int grid_cam_x  = camera.nx/block_cam_x + (camera.nx%block_cam_x > 0);
    const int grid_cam_y  = camera.ny/block_cam_y + (camera.ny%block_cam_y > 0);

    dim3 grid_cam(grid_cam_x, grid_cam_y);
    dim3 block_cam(block_cam_x, block_cam_y);

    count_to_flux_2d<<<grid_cam, block_cam>>>(
            camera, photons_per_pixel,
            toa_src,
            toa_factor,
            camera_count.ptr(),
            flux_camera.ptr());

}

void Raytracer_bw::trace_rays_bb(
        const int igpt,
        const int photons_per_pixel,
        const int nlay,
        const Vector<int>& grid_cells,
        const Vector<Float>& grid_d,
        const Vector<int>& kn_grid,
        const Array_gpu<Float,1>& z_lev,
        const Array_gpu<Float,3>& mie_cdf,
        const Array_gpu<Float,4>& mie_ang,
        const Array_gpu<Float,4>& mie_phase,
        const Array_gpu<Float,1>& mie_phase_ang,
        const Array_gpu<Float,2>& r_eff,
        const Array_gpu<Float,2>& tau_total,
        const Array_gpu<Float,2>& ssa_total,
        const Array_gpu<Float,2>& tau_cloud,
        const Array_gpu<Float,2>& ssa_cloud,
        const Array_gpu<Float,2>& asy_cloud,
        const Array_gpu<Float,2>& tau_aeros,
        const Array_gpu<Float,2>& ssa_aeros,
        const Array_gpu<Float,2>& asy_aeros,
        const Array_gpu<Float,2>& surface_albedo,
        const Array_gpu<Float,1>& land_use_map,
        const Float zenith_angle,
        const Float azimuth_angle,
        const Float toa_src,
        const Camera& camera,
        Array_gpu<Float,2>& flux_camera)
{
    const Float mu = std::abs(std::cos(zenith_angle));

    // set of block and grid dimensions used in data processing kernels - requires some proper tuning later
    const int block_col_x = 8;
    const int block_col_y = 8;
    const int block_z = 4;

    const int grid_col_x  = grid_cells.x/block_col_x + (grid_cells.x%block_col_x > 0);
    const int grid_col_y  = grid_cells.y/block_col_y + (grid_cells.y%block_col_y > 0);
    const int grid_z  = grid_cells.z/block_z + (grid_cells.z%block_z > 0);

    dim3 grid_3d(grid_col_x, grid_col_y, grid_z);
    dim3 block_3d(block_col_x, block_col_y, block_z);

    // bundle optical properties in struct
    Array_gpu<Float,3> k_ext({grid_cells.x, grid_cells.y, grid_cells.z});
    Array_gpu<Optics_scat,3> ssa_asy({grid_cells.x, grid_cells.y, grid_cells.z});

    bundles_optical_props_bb<<<grid_3d, block_3d>>>(
            grid_cells, nlay, grid_d.z,
            tau_total.ptr(), ssa_total.ptr(),
            tau_cloud.ptr(), ssa_cloud.ptr(), asy_cloud.ptr(),
            tau_aeros.ptr(), ssa_aeros.ptr(), asy_aeros.ptr(),
            k_ext.ptr(), ssa_asy.ptr());

    // create k_null_grid
    const int block_kn_x = 2;
    const int block_kn_y = 2;
    const int block_kn_z = 4;

    const int grid_kn_x  = kn_grid.x/block_kn_x + (kn_grid.x%block_kn_x > 0);
    const int grid_kn_y  = kn_grid.y/block_kn_y + (kn_grid.y%block_kn_y > 0);
    const int grid_kn_z  = kn_grid.z/block_kn_z + (kn_grid.z%block_kn_z > 0);

    dim3 grid_kn(grid_kn_x, grid_kn_y, grid_kn_z);
    dim3 block_kn(block_kn_x, block_kn_y, block_kn_z);

    Array_gpu<Grid_knull,3> k_null_grid({kn_grid.x, kn_grid.y, kn_grid.z});
    const Float k_ext_null_min = Float(1e-3);

    create_knull_grid<<<grid_kn, block_kn>>>(
            grid_cells, kn_grid, k_ext_null_min,
            k_ext.ptr(), k_null_grid.ptr());

    // TOA-TOD profile (at x=0, y=0)
    const int nbg = nlay - grid_cells.z;
    Array_gpu<Float,1> k_ext_bg({nbg});
    Array_gpu<Optics_scat,1> ssa_asy_bg({nbg});
    Array_gpu<Float,1> z_lev_bg({nbg+1});

    const int block_1d_z = 16;
    const int grid_1d_z  = nbg/block_1d_z + (nbg%block_1d_z > 0);
    dim3 grid_1d(grid_1d_z);
    dim3 block_1d(block_1d_z);

    background_profile_bb<<<grid_1d, block_1d>>>(
            grid_cells, nbg, z_lev.ptr(),
            tau_total.ptr(), ssa_total.ptr(),
            tau_cloud.ptr(), ssa_cloud.ptr(), asy_cloud.ptr(),
            tau_aeros.ptr(), ssa_aeros.ptr(), asy_aeros.ptr(),
            k_ext_bg.ptr(), ssa_asy_bg.ptr(), z_lev_bg.ptr());

    Array_gpu<Float,2> camera_count({camera.nx, camera.ny});
    Array_gpu<Float,2> shot_count({camera.nx, camera.ny});
    Array_gpu<int,1> counter({1});

    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(camera.nx, camera.ny, camera_count.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(camera.nx, camera.ny, shot_count.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(1, counter.ptr());

    // domain sizes
    const Vector<Float> grid_size = grid_d * grid_cells;

    // direction of direct sun rays
    const Vector<Float> sun_direction = {std::sin(zenith_angle) * std::sin(azimuth_angle),
                                         std::sin(zenith_angle) * std::cos(azimuth_angle),
                                         std::cos(zenith_angle)};

    dim3 grid{bw_kernel_grid}, block{bw_kernel_block};

    const int mie_table_size = mie_cdf.size();

    ray_tracer_kernel_bw<<<grid, block, nbg*sizeof(Float)+ 2 * sizeof(Float)*mie_table_size>>>(
            igpt-1,
            photons_per_pixel, k_null_grid.ptr(),
            camera_count.ptr(),
            shot_count.ptr(),
            counter.ptr(),
            k_ext.ptr(), ssa_asy.ptr(),
            k_ext_bg.ptr(), ssa_asy_bg.ptr(),
            z_lev_bg.ptr(),
            r_eff.ptr(),
            surface_albedo.ptr(),
            land_use_map.ptr(),
            mu,
            grid_size, grid_d, grid_cells, kn_grid,
            sun_direction, camera, nbg,
            mie_cdf.ptr(), mie_ang.ptr(),
            mie_phase.ptr(), mie_phase_ang.ptr(),
            mie_table_size);

    //// convert counts to fluxes
    const int block_cam_x = 8;
    const int block_cam_y = 8;

    const int grid_cam_x  = camera.nx/block_cam_x + (camera.nx%block_cam_x > 0);
    const int grid_cam_y  = camera.ny/block_cam_y + (camera.ny%block_cam_y > 0);

    dim3 grid_cam(grid_cam_x, grid_cam_y);
    dim3 block_cam(block_cam_x, block_cam_y);


    count_to_flux_2d<<<grid_cam, block_cam>>>(
            camera, photons_per_pixel,
            toa_src,
            sun_solid_angle_reciprocal,
            camera_count.ptr(),
            flux_camera.ptr());

}
void Raytracer_bw::accumulate_clouds(
        const Vector<Float>& grid_d,
        const Vector<int>& grid_cells,
        const Array_gpu<Float,2>& lwp,
        const Array_gpu<Float,2>& iwp,
        const Array_gpu<Float,2>& tau_cloud,
        const Camera& camera,
        Array_gpu<Float,2>& liwp_cam,
        Array_gpu<Float,2>& tauc_cam,
        Array_gpu<Float,2>& dist_cam,
        Array_gpu<Float,2>& zen_cam)
{
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(camera.nx, camera.ny, liwp_cam.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(camera.nx, camera.ny, tauc_cam.ptr());
    Gas_optics_rrtmgp_kernels_cuda_rt::zero_array(camera.nx, camera.ny, dist_cam.ptr());
    
    const int n_pix = camera.nx * camera.ny;
    const int n_block = std::min(n_pix, 512);
    const int n_grid = std::ceil(Float(n_pix)/n_block);

    dim3 grid(n_grid);
    dim3 block(n_block);

    // domain sizes
    const Vector<Float> grid_size = grid_d * grid_cells;
    
    accumulate_clouds_kernel<<<grid, block>>>(
        lwp.ptr(),
        iwp.ptr(),
        tau_cloud.ptr(),
        grid_d,
        grid_size,
        grid_cells,
        liwp_cam.ptr(),
        tauc_cam.ptr(),
        dist_cam.ptr(),
        zen_cam.ptr(),
        camera);

}


            
            
            
            
            
            
            
            
            
