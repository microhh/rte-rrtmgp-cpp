#include <chrono>

#include "rte_solver_kernels_cuda_rt.h"
#include "tools_gpu.h"
#include "Array.h"
#include "tuner.h"

#include <iomanip>


namespace
{
    #include "rte_solver_kernels_rt.cu"
    
    using Tools_gpu::calc_grid_size;
}


namespace Rte_solver_kernels_cuda_rt
{
    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
                  const Float* inc_flux_dir, const Float* mu0, Float* gpt_flux_dir)
    {
        const int block_col = 32;
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col);
        dim3 block_gpu(block_col);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dir, mu0, gpt_flux_dir);
    }


    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, Float* gpt_flux_dn)
    {
        const int block_col = 32;
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col);
        dim3 block_gpu(block_col);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, gpt_flux_dn);
    }


    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const Float* inc_flux_dif, Float* gpt_flux_dn)
    {
        const int block_col = 32;
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col);
        dim3 block_gpu(block_col);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dif, gpt_flux_dn);
    }

    void lw_solver_noscat_gaussquad(
            const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const int nmus,
            const Float* ds, const Float* weights, const Float* tau, const Float* lay_source,
            const Float* lev_source_inc, const Float* lev_source_dec, const Float* sfc_emis,
            const Float* sfc_src, Float* flux_up, Float* flux_dn,
            const Float* sfc_src_jac, Float* flux_up_jac)
    {
        Tuner_map& tunings = Tuner::get_map();
        
        Float eps = std::numeric_limits<Float>::epsilon();

        const int flx_size = ncol*(nlay+1);
        const int opt_size = ncol*nlay;
        const int sfc_size = ncol;
    
        Float* source_sfc = Tools_gpu::allocate_gpu<Float>(sfc_size);
        Float* source_sfc_jac = Tools_gpu::allocate_gpu<Float>(sfc_size);
        Float* sfc_albedo = Tools_gpu::allocate_gpu<Float>(sfc_size);
        Float* tau_loc = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* trans = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_dn = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_up = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* radn_dn = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* radn_up = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* radn_up_jac = Tools_gpu::allocate_gpu<Float>(flx_size);

        const int block_col1d = 64;
        const int grid_col1d = ncol/block_col1d + (ncol%block_col1d > 0);

        dim3 grid_gpu1d(grid_col1d, 1, 1);
        dim3 block_gpu1d(block_col1d, 1, 1);

        const int block_col2d = 96;
        const int block_lay2d = 1;

        const int grid_col2d = ncol/block_col2d + (ncol%block_col2d > 0);
        const int grid_lay2d = (nlay+1)/block_lay2d + ((nlay+1)%block_lay2d > 0);

        dim3 grid_gpu2d(grid_col2d, grid_lay2d, 1);
        dim3 block_gpu2d(block_col2d, block_lay2d, 1);

        const int top_level = top_at_1 ? 0 : nlay;

        // Step 1.
        dim3 grid_1, block_1;

        if (tunings.count("lw_step_1_rt") == 0)
        {
            Float* flux_up_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);
            Float* flux_dn_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);
            Float* flux_up_jac_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);
 
            std::tie(grid_1, block_1) = tune_kernel(
                    "lw_step_1_rt",
                    dim3(ncol, nlay, 1), 
                    {8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024}, {1}, {1},
                    lw_solver_noscat_step_1_kernel,
                    ncol, nlay, ngpt, eps, top_at_1, ds, weights, tau, lay_source,
                    lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up_tmp, flux_dn_tmp, sfc_src_jac,
                    flux_up_jac_tmp, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

            Tools_gpu::free_gpu<Float>(flux_up_tmp);
            Tools_gpu::free_gpu<Float>(flux_dn_tmp);
            Tools_gpu::free_gpu<Float>(flux_up_jac_tmp);

            tunings["lw_step_1_rt"].first = grid_1;
            tunings["lw_step_1_rt"].second = block_1;
        
        }
        else
        {
            block_1 = tunings["lw_step_1_rt"].second;
        }

        grid_1 = calc_grid_size(block_1, dim3(ncol, nlay, 1));

        lw_solver_noscat_step_1_kernel<<<grid_1, block_1>>>(
                ncol, nlay, ngpt, eps, top_at_1, ds, weights, tau, lay_source,
                lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
                flux_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);


        // Step 2.
        dim3 grid_2, block_2;

        if (tunings.count("lw_step_2_rt") == 0)
        {
            Float* flux_up_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);
            Float* flux_dn_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);
            Float* flux_up_jac_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);

            std::tie(grid_2, block_2) = tune_kernel(
                    "lw_step_2_rt",
                    dim3(ncol, 1, 1), 
                    {64, 128, 256, 384, 512, 768, 1024}, {1}, {1},
                    lw_solver_noscat_step_2_kernel,
                    ncol, nlay, ngpt, eps, top_at_1, ds, weights, tau, lay_source,
                    lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up_tmp, flux_dn_tmp, sfc_src_jac,
                    flux_up_jac_tmp, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

            Tools_gpu::free_gpu<Float>(flux_up_tmp);
            Tools_gpu::free_gpu<Float>(flux_dn_tmp);
            Tools_gpu::free_gpu<Float>(flux_up_jac_tmp);

            tunings["lw_step_2_rt"].first = grid_2;
            tunings["lw_step_2_rt"].second = block_2;
        }
        else
        {
            block_2 = tunings["lw_step_2_rt"].second;
        }
        
        grid_2 = calc_grid_size(block_2, dim3(ncol, 1, 1));

        lw_solver_noscat_step_2_kernel<<<grid_2, block_2>>>(
                ncol, nlay, ngpt, eps, top_at_1, ds, weights, tau, lay_source,
                lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
                flux_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);


        // Step 3.
        dim3 grid_3, block_3;

        if (tunings.count("lw_step_3_rt") == 0)
        {
            Float* flux_up_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);
            Float* flux_dn_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);
            Float* flux_up_jac_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);
            
            std::tie(grid_3, block_3) = tune_kernel(
                    "lw_step_3_rt",
                    dim3(ncol, nlay+1, 1), 
                    {8, 16, 24, 32, 48, 64, 96, 128, 256}, {1, 2, 4, 8}, {1},
                    lw_solver_noscat_step_3_kernel,
                    ncol, nlay, ngpt, eps, top_at_1, ds, weights, tau, lay_source,
                    lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up_tmp, flux_dn_tmp, sfc_src_jac,
                    flux_up_jac_tmp, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

            Tools_gpu::free_gpu<Float>(flux_up_tmp);
            Tools_gpu::free_gpu<Float>(flux_dn_tmp);
            Tools_gpu::free_gpu<Float>(flux_up_jac_tmp);
            
            tunings["lw_step_3_rt"].first = grid_3;
            tunings["lw_step_3_rt"].second = block_3;
        }
        else
        {
            block_3 = tunings["lw_step_3_rt"].second;
        }
        
        grid_3 = calc_grid_size(block_3, dim3(ncol, nlay+1, 1));

        lw_solver_noscat_step_3_kernel<<<grid_3, block_3>>>(
                ncol, nlay, ngpt, eps, top_at_1, ds, weights, tau, lay_source,
                lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
                flux_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

        apply_BC_kernel_lw<<<grid_gpu1d, block_gpu1d>>>(top_level, ncol, nlay, ngpt, top_at_1, flux_dn, radn_dn);

        if (nmus > 1)
        {
            for (int imu=1; imu<nmus; ++imu)
            {
                throw std::runtime_error("Not implemented due to lacking test case");
                /*
                lw_solver_noscat_step_1_kernel<<<grid_1, block_1>>>(
                        ncol, nlay, ngpt, eps, top_at_1, ds+imu, weights+imu, tau, lay_source,
                        lev_source_inc, lev_source_dec, sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac,
                        radn_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

                lw_solver_noscat_step_2_kernel<<<grid_2, block_2>>>(
                        ncol, nlay, ngpt, eps, top_at_1, ds+imu, weights+imu, tau, lay_source,
                        lev_source_inc, lev_source_dec, sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac,
                        radn_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

                lw_solver_noscat_step_3_kernel<<<grid_3, block_3>>>(
                        ncol, nlay, ngpt, eps, top_at_1, ds+imu, weights+imu, tau, lay_source,
                        lev_source_inc, lev_source_dec, sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac,
                        radn_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

                add_fluxes_kernel<<<grid_gpu2d, block_gpu2d>>>(
                        ncol, nlay+1, ngpt,
                        radn_up, radn_dn, radn_up_jac,
                        flux_up, flux_dn, flux_up_jac);
                */
            }
        }
    
        Tools_gpu::free_gpu(source_sfc);
        Tools_gpu::free_gpu(source_sfc_jac);
        Tools_gpu::free_gpu(sfc_albedo);
        Tools_gpu::free_gpu(tau_loc);
        Tools_gpu::free_gpu(trans);
        Tools_gpu::free_gpu(source_dn);
        Tools_gpu::free_gpu(source_up);
        Tools_gpu::free_gpu(radn_dn);
        Tools_gpu::free_gpu(radn_up);
        Tools_gpu::free_gpu(radn_up_jac);
    }


    void sw_solver_2stream(const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
                           const Float* tau, const Float* ssa, const Float* g,
                           const Float* mu0, const Float* sfc_alb_dir, const Float* sfc_alb_dif,
                           Float* flux_up, Float* flux_dn, Float* flux_dir)
    {
        Tuner_map& tunings = Tuner::get_map();

        const int flx_size = ncol*(nlay+1);
        const int opt_size = ncol*nlay;
        const int sfc_size = ncol;

        Float* r_dif = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* t_dif = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* r_dir = nullptr;
        Float* t_dir = nullptr;
        Float* t_noscat = nullptr;
        Float* source_up = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_dn = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_sfc = Tools_gpu::allocate_gpu<Float>(sfc_size);
        Float* albedo = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* src = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* denom = Tools_gpu::allocate_gpu<Float>(opt_size);

        dim3 grid_source{ncol, 1}, block_source;


        // Step 1.
        if (tunings.count("sw_source_2stream_kernel_rt") == 0)
        {
            if (top_at_1)
            {
                std::tie(grid_source, block_source) = tune_kernel(
                        "sw_source_2stream_kernel_rt",
                        dim3(ncol, 1), 
                        {32, 64, 96, 128, 256, 384, 512, 768, 1024}, {1}, {1},
                        sw_source_2stream_kernel<1>,
                        ncol, nlay, ngpt, tau, ssa, g, mu0, r_dif, t_dif,
                        sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir);
            }
            else
            {
                std::tie(grid_source, block_source) = tune_kernel(
                        "sw_source_2stream_kernel_rt",
                        dim3(ncol, 1), 
                        {32, 64, 96, 128, 256, 384, 512, 768, 1024}, {1}, {1},
                        sw_source_2stream_kernel<0>,
                        ncol, nlay, ngpt, tau, ssa, g, mu0, r_dif, t_dif,
                        sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir);
            }

            tunings["sw_source_2stream_kernel_rt"].first = grid_source;
            tunings["sw_source_2stream_kernel_rt"].second = block_source;
        }
        else
        {
            block_source = tunings["sw_source_2stream_kernel_rt"].second;
        }
            
        grid_source = calc_grid_size(block_source, dim3(ncol, 1));
        
        if (top_at_1)
        {
            sw_source_2stream_kernel<1><<<grid_source, block_source>>>(
                    ncol, nlay, ngpt, tau, ssa, g, mu0, r_dif, t_dif,
                    sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir);
        }
        else
        {
            sw_source_2stream_kernel<0><<<grid_source, block_source>>>(
                    ncol, nlay, ngpt, tau, ssa, g, mu0, r_dif, t_dif,
                    sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir);
        }


        // Step 2.
        dim3 grid_adding, block_adding;

        if (tunings.count("sw_adding_rt") == 0)
        {
            Float* flux_up_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);
            Float* flux_dn_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);
            Float* flux_dir_tmp = Tools_gpu::allocate_gpu<Float>((nlay+1)*ncol);

            if (top_at_1)
            {
                std::tie(grid_adding, block_adding) = tune_kernel(
                        "sw_adding_rt",
                        dim3(ncol, 1),
                        {16, 32, 64, 96, 128, 256, 384, 512}, {1}, {1},
                        sw_adding_kernel<1>,
                        ncol, nlay, ngpt, top_at_1,
                        sfc_alb_dif, r_dif, t_dif,
                        source_dn, source_up, source_sfc,
                        flux_up_tmp, flux_dn_tmp, flux_dir_tmp, albedo, src, denom);
            }
            else
            {
                std::tie(grid_adding, block_adding) = tune_kernel(
                        "sw_adding_rt",
                        dim3(ncol, 1), 
                        {16, 32, 64, 96, 128, 256, 384, 512}, {1}, {1},
                        sw_adding_kernel<0>,
                        ncol, nlay, ngpt, top_at_1,
                        sfc_alb_dif, r_dif, t_dif,
                        source_dn, source_up, source_sfc,
                        flux_up_tmp, flux_dn_tmp, flux_dir_tmp, albedo, src, denom);
            }
            
            Tools_gpu::free_gpu<Float>(flux_up_tmp);
            Tools_gpu::free_gpu<Float>(flux_dn_tmp);
            Tools_gpu::free_gpu<Float>(flux_dir_tmp);
            tunings["sw_adding_rt"].first = grid_adding;
            tunings["sw_adding_rt"].second = block_adding;
        }
        else
        {
            block_adding = tunings["sw_adding_rt"].second;
        }
        
        grid_adding = calc_grid_size(block_adding, dim3(ncol, 1));

        if (top_at_1)
        {
            sw_adding_kernel<1><<<grid_adding, block_adding>>>(
                ncol, nlay, ngpt, top_at_1,
                sfc_alb_dif, r_dif, t_dif,
                source_dn, source_up, source_sfc,
                flux_up, flux_dn, flux_dir, albedo, src, denom);
        }
        else
        {
            sw_adding_kernel<0><<<grid_adding, block_adding>>>(
                        ncol, nlay, ngpt, top_at_1,
                        sfc_alb_dif, r_dif, t_dif,
                        source_dn, source_up, source_sfc,
                        flux_up, flux_dn, flux_dir, albedo, src, denom);
        }

        Tools_gpu::free_gpu(r_dif);
        Tools_gpu::free_gpu(t_dif);
        Tools_gpu::free_gpu(source_up);
        Tools_gpu::free_gpu(source_dn);
        Tools_gpu::free_gpu(source_sfc);
        Tools_gpu::free_gpu(albedo);
        Tools_gpu::free_gpu(src);
        Tools_gpu::free_gpu(denom);
    }
}

