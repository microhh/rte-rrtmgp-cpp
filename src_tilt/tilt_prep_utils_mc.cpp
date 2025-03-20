#include <boost/algorithm/string.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_profiler_api.h>

#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "Gas_concs.h"
#include "tilted_column.h"
#include "tilt_prep_utils.h"
#include "tilt_prep_utils_mc.h"
#include "types.h"
#include "profiler.h"


void tilt_fields_single_column(const int idx_col_x, const int idx_col_y,
    const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1> zh, const Array<Float,1> z,
    const Array<Float,1> zh_tilt, const Array<ijk,1> path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Array<Float,2>* h2o_copy, Array<Float,2>* o3_copy, 
    const bool switch_cloud_optics, const bool switch_liq_cloud_optics, const bool switch_ice_cloud_optics
) {
    // Cloud optics section
    auto start_cloud_total = std::chrono::high_resolution_clock::now();
    
    if (switch_cloud_optics)
    {
        // Liquid cloud processing
        if (switch_liq_cloud_optics)
        {
            auto start_liquid = std::chrono::high_resolution_clock::now();
            
            create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), lwp_copy->v());
            create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), rel_copy->v());

            lwp_copy->expand_dims({1, n_z_tilt});
            rel_copy->expand_dims({1, n_z_tilt});
            
            auto end_liquid = std::chrono::high_resolution_clock::now();
            Profiler::instance().add_measurement("Cloud optics - liquid tilted columns",
                std::chrono::duration<double, std::milli>(end_liquid - start_liquid).count());
        }
        
        // Ice cloud processing
        if (switch_ice_cloud_optics)
        {
            auto start_ice = std::chrono::high_resolution_clock::now();
            
            create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), iwp_copy->v());
            create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), dei_copy->v());

            iwp_copy->expand_dims({1, n_z_tilt});
            dei_copy->expand_dims({1, n_z_tilt});
            
            auto end_ice = std::chrono::high_resolution_clock::now();
            Profiler::instance().add_measurement("Cloud optics - ice tilted columns",
                std::chrono::duration<double, std::milli>(end_ice - start_ice).count());
        }

        // Denormalization
        auto start_denorm = std::chrono::high_resolution_clock::now();
        
        for (int ilay = 1; ilay <= n_z_tilt; ++ilay)    
        {
            Float dz = zh_tilt({ilay + 1}) - zh_tilt({ilay});
            if (switch_liq_cloud_optics)
            {
                (*lwp_copy)({1, ilay}) *= dz;
            }
            if (switch_ice_cloud_optics)
            {
                (*iwp_copy)({1, ilay}) *= dz;
            }
        }
        
        auto end_denorm = std::chrono::high_resolution_clock::now();
        Profiler::instance().add_measurement("Cloud optics - denormalization by dz",
            std::chrono::duration<double, std::milli>(end_denorm - start_denorm).count());
    }
    
    auto end_cloud_total = std::chrono::high_resolution_clock::now();
    Profiler::instance().add_measurement("Cloud optics total",
        std::chrono::duration<double, std::milli>(end_cloud_total - start_cloud_total).count());

    // Gas processing section
    auto start_gas_total = std::chrono::high_resolution_clock::now();
    create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), h2o_copy->v());
    create_single_tilted_columns(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), o3_copy->v());    
    
    auto end_gas_total = std::chrono::high_resolution_clock::now();
    Profiler::instance().add_measurement("Gases processing total",
        std::chrono::duration<double, std::milli>(end_gas_total - start_gas_total).count());

    // Pressure and temperature section
    auto start_pt_total = std::chrono::high_resolution_clock::now();
    
    // Temperature processing
    auto start_temp = std::chrono::high_resolution_clock::now();
    create_single_tilted_columns_levlay(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), t_lay_copy->v(), t_lev_copy->v());
    t_lay_copy->expand_dims({1, n_z_tilt});
    t_lev_copy->expand_dims({1, n_zh_tilt});
    auto end_temp = std::chrono::high_resolution_clock::now();
    Profiler::instance().add_measurement("Temperature tilted columns",
        std::chrono::duration<double, std::milli>(end_temp - start_temp).count());
    
    // Pressure processing
    auto start_press = std::chrono::high_resolution_clock::now();
    create_single_tilted_columns_levlay(idx_col_x, idx_col_y, n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), p_lay_copy->v(), p_lev_copy->v());
    p_lay_copy->expand_dims({1, n_z_tilt});
    p_lev_copy->expand_dims({1, n_zh_tilt});
    auto end_press = std::chrono::high_resolution_clock::now();
    Profiler::instance().add_measurement("Pressure tilted columns",
        std::chrono::duration<double, std::milli>(end_press - start_press).count());
    
    auto end_pt_total = std::chrono::high_resolution_clock::now();
    Profiler::instance().add_measurement("Pressure and temp total",
        std::chrono::duration<double, std::milli>(end_pt_total - start_pt_total).count());
}

void compress_fields_single_column(const int compress_lay_start_idx, const int n_col_x, const int n_col_y,
    const int n_z_in, const int n_zh_in,  const int n_z_tilt,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Array<Float,2>* h2o_copy, Array<Float,2>* o3_copy, 
    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics)
{
    const int n_col = n_col_x*n_col_y;

    if (switch_liq_cloud_optics)
    {
        compress_columns_weighted_avg(n_col_x, n_col_y, 
                                        n_z_in, n_z_tilt, compress_lay_start_idx, 
                                        rel_copy->v(), lwp_copy->v());
        rel_copy->expand_dims({n_col, n_z_in});

        compress_columns(n_col_x, n_col_y, 
                            n_z_in, n_z_tilt,
                            compress_lay_start_idx, lwp_copy->v());
        lwp_copy->expand_dims({n_col, n_z_in}); 
    }
    if (switch_ice_cloud_optics)
    {
        compress_columns_weighted_avg(n_col_x, n_col_y, 
                                        n_z_in, n_z_tilt, compress_lay_start_idx, 
                                        dei_copy->v(), iwp_copy->v());
        dei_copy->expand_dims({n_col, n_z_in});
        
        compress_columns(n_col_x, n_col_y, 
                            n_z_in, n_z_tilt,
                            compress_lay_start_idx, iwp_copy->v());
        iwp_copy->expand_dims({n_col, n_z_in}); 
    }

    // GASES 
    compress_columns_weighted_avg(n_col_x, n_col_y,
                                            n_z_in, n_z_tilt,
                                            compress_lay_start_idx, 
                                            h2o_copy->v(), 
                                            p_lay_copy->v());
    h2o_copy->expand_dims({n_col, n_z_in});
    compress_columns_weighted_avg(n_col_x, n_col_y,
                                            n_z_in, n_z_tilt,
                                            compress_lay_start_idx, 
                                            o3_copy->v(), 
                                            p_lay_copy->v());
    o3_copy->expand_dims({n_col, n_z_in});

    // PRESSURE AND TEMPERATURE
    compress_columns_p_or_t(n_col_x, n_col_y, n_z_in, n_z_tilt,
                            compress_lay_start_idx, 
                            p_lev_copy->v(), p_lay_copy->v());
    p_lay_copy->expand_dims({n_col, n_z_in});
    p_lev_copy->expand_dims({n_col, n_zh_in});
    compress_columns_p_or_t(n_col_x, n_col_y, n_z_in, n_z_tilt,
                            compress_lay_start_idx, 
                            t_lev_copy->v(), t_lay_copy->v());
    t_lay_copy->expand_dims({n_col, n_z_in});
    t_lev_copy->expand_dims({n_col, n_zh_in});
}



void restore_bkg_profile_bundle(const int n_col_x, const int n_col_y, 
    const int n_lay, const int n_lev, 
    const int n_lay_tot, const int n_lev_tot, 
    const int n_z_in, const int n_zh_in,
    const int bkg_start_z, const int bkg_start_zh, 
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, 
    Gas_concs& gas_concs_copy,
    Array<Float,2>* p_lay, Array<Float,2>* t_lay, Array<Float,2>* p_lev, Array<Float,2>* t_lev, 
    Array<Float,2>* lwp, Array<Float,2>* iwp, Array<Float,2>* rel, Array<Float,2>* dei, 
    Gas_concs& gas_concs, 
    std::vector<std::string> gas_names,
    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics
)
{
    const int n_col = n_col_x*n_col_y;

    if (switch_liq_cloud_optics)
    {
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, lwp_copy->v(), lwp->v());
        lwp_copy->expand_dims({n_col, n_lay_tot});
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, rel_copy->v(), rel->v());
        rel_copy->expand_dims({n_col, n_lay_tot});
    }
    if (switch_ice_cloud_optics)
    {
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, iwp_copy->v(), iwp->v());
        iwp_copy->expand_dims({n_col, n_lay_tot});
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, dei_copy->v(), dei->v());
        dei_copy->expand_dims({n_col, n_lay_tot});
    }

    for (const auto& gas_name : gas_names) {
        if (!gas_concs_copy.exists(gas_name)) {
            continue;
        }
        const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);
        const Array<Float,2>& gas_full = gas_concs.get_vmr(gas_name);

        std::string var_name = "vmr_" + gas_name;
        if (gas.size() > 1) {
            std::vector<Float> gas_copy = gas.v();
            std::vector<Float> gas_full_copy = gas_full.v();
            restore_bkg_profile(n_col_x, n_col_y, 
                                n_lay, n_z_in, 
                                bkg_start_z, 
                                gas_copy, gas_full_copy);
            
            Array<Float,2> gas_tmp({n_col, n_lay_tot});
            gas_tmp = std::move(gas_copy);
            gas_tmp.expand_dims({n_col, n_lay_tot});
            
            gas_concs_copy.set_vmr(gas_name, gas_tmp);
            
        }
    }

    restore_bkg_profile(n_col_x, n_col_y, 
                        n_lay, n_z_in, 
                        bkg_start_z, 
                        p_lay_copy->v(), p_lay->v());
    p_lay_copy->expand_dims({n_col, n_lay_tot});

    restore_bkg_profile(n_col_x, n_col_y, 
                        n_lay, n_z_in, 
                        bkg_start_z, 
                        t_lay_copy->v(), t_lay->v());
    t_lay_copy->expand_dims({n_col, n_lay_tot});

    restore_bkg_profile(n_col_x, n_col_y, 
                        n_lev, n_zh_in, 
                        bkg_start_zh, 
                        p_lev_copy->v(), p_lev->v());
    p_lev_copy->expand_dims({n_col, n_lev_tot});
    restore_bkg_profile(n_col_x, n_col_y, 
                        n_lev, n_zh_in, 
                        bkg_start_zh, 
                        t_lev_copy->v(), t_lev->v());
    t_lev_copy->expand_dims({n_col, n_lev_tot});
}


void post_process_output(const std::vector<ColumnResult>& col_results,
                         const int n_col_x, const int n_col_y,
                         const int n_z, const int n_zh,
                         Array<Float,2>* p_lay_out,
                         Array<Float,2>* t_lay_out,
                         Array<Float,2>* p_lev_out,
                         Array<Float,2>* t_lev_out,
                         Array<Float,2>* lwp_out,
                         Array<Float,2>* iwp_out,
                         Array<Float,2>* rel_out,
                         Array<Float,2>* dei_out,
                         Gas_concs& gas_concs_out,
                         const std::vector<std::string>& gas_names,
                         const bool switch_cloud_optics,
                         const bool switch_liq_cloud_optics,
                         const bool switch_ice_cloud_optics)
{
    const int total_cols = n_col_x * n_col_y;
    const int stride = total_cols;
    for (int idx_y = 0; idx_y < n_col_y; ++idx_y) {
        for (int idx_x = 0; idx_x < n_col_x; ++idx_x) {
            int col_idx = idx_x + idx_y * n_col_x;
            const ColumnResult& col = col_results[col_idx];
            int base_idx = col_idx;

            for (int j = 0; j < n_z; ++j) {
                int out_idx = base_idx + j * stride;
                p_lay_out->v()[out_idx] = col.p_lay.v()[j];
                t_lay_out->v()[out_idx] = col.t_lay.v()[j];
                if (switch_liq_cloud_optics) {
                    lwp_out->v()[out_idx] = col.lwp.v()[j];
                    rel_out->v()[out_idx] = col.rel.v()[j];
                }
                if (switch_ice_cloud_optics) {
                    iwp_out->v()[out_idx] = col.iwp.v()[j];
                    dei_out->v()[out_idx] = col.dei.v()[j];
                }
            }

            for (int j = 0; j < n_zh; ++j) {
                int out_idx = base_idx + j * stride;
                p_lev_out->v()[out_idx] = col.p_lev.v()[j];
                t_lev_out->v()[out_idx] = col.t_lev.v()[j];
            }
            
            Array<Float,2>& h2o_dest = const_cast<Array<Float,2>&>(gas_concs_out.get_vmr("h2o"));
            for (int j = 0; j < n_z; ++j) {
                const int out_idx = base_idx + j * stride;
                h2o_dest.v()[out_idx] = col.h2o.v()[j];
            }

            Array<Float,2>& o3_dest = const_cast<Array<Float,2>&>(gas_concs_out.get_vmr("o3"));
            for (int j = 0; j < n_z; ++j) {
                const int out_idx = base_idx + j * stride;
                o3_dest.v()[out_idx] = col.o3.v()[j];
            }

        }
    }
}