#include <cmath>
#include <numeric>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include "Gas_concs.h"
#include "Netcdf_interface.h"
#include "Gas_optics_nn.h"
#include "Array.h"
#include "Status.h"
#include "Optical_props.h"
#include "Source_functions.h"
#include "rrtmgp_kernels.h"
#include <time.h>
#include <sys/time.h>

#define restrict __restrict__

namespace
{
    inline float logarithm(float x)
    {
        x = std::sqrt(x);
        x = std::sqrt(x);
        x = std::sqrt(x);
        x = std::sqrt(x);
        x = (x-1.0f) * 16.0f;
        return x;
    }
 
    template<typename TF>
    void copy_arrays_tau(
                 const float* restrict const data_in,
                 const float* restrict const data_dp,
                 TF* restrict const data_out,
                 const int n_col, const int n_bot,
                 const int n_top, const int n_gpt,
                 const int n_lay)
    {
        const float* dp_temp = &data_dp[n_col*n_bot];
        const int n_sub = n_top-n_bot;
        for (int i=0; i<n_gpt; ++i)
        {
            const int outidx = i*n_lay*n_col + n_bot*n_col;
            const float* in_temp = &data_in[i*n_sub*n_col];
            TF* out_temp = &data_out[outidx];

            #pragma ivdep            
            for (int j=0; j<n_col*n_sub; ++j)
                out_temp[j] = in_temp[j] * dp_temp[j];
        }
    }

    template<typename TF>
    void copy_arrays_ssa(
                 const float* restrict const data_in,
                 TF* restrict const data_out,
                 const int n_col, const int n_bot,
                 const int n_top, const int n_gpt,
                 const int nlay)
    {
        const int n_sub = n_top-n_bot;
        for (int i=0; i<n_gpt; ++i)
        {
            const int outidx = i*nlay*n_col+n_bot*n_col;
            const float* in_temp = &data_in[i*n_sub*n_col];
            TF* out_temp = &data_out[outidx];
            #pragma ivdep            
            for (int j=0; j<n_col*n_sub; ++j)
            {
                out_temp[j] = in_temp[j]; 
            }             
        }
    }

    template<typename TF>
    void copy_arrays_plk(
                 const float* restrict const data_in,
                 TF* restrict const data_out1,
                 TF* restrict const data_out2,
                 TF* restrict const data_out3,
                 const int n_col, const int n_bot,
                 const int n_top, const int n_gpt,
                 const int nlay)
    {
        const int n_sub = n_top-n_bot;
        for (int i=0; i<n_gpt; ++i)
        {
            const int outidx = i*nlay*n_col+n_bot*n_col;
            const float* in_temp1 = &data_in[i*n_sub*n_col];
            const float* in_temp2 = &data_in[(i+n_gpt)*n_sub*n_col];
            const float* in_temp3 = &data_in[(i+n_gpt+n_gpt)*n_sub*n_col];
            TF* out_temp1= &data_out1[outidx];
            TF* out_temp2= &data_out2[outidx];
            TF* out_temp3= &data_out3[outidx];
            #pragma ivdep           
            for (int j=0; j<n_col*n_sub; ++j)
            {
                out_temp1[j] = in_temp1[j];
                out_temp2[j] = in_temp2[j];
                out_temp3[j] = in_temp3[j];
            }
        }
    }
}
       
template<typename TF>
Gas_optics_nn<TF>::Gas_optics_nn(
        const Array<std::string,1>& gas_names,
        const Array<int,2>& band2gpt,
        const Array<TF,2>& band_lims_wavenum,
        const std::string& file_name_weights,
        Netcdf_file& input_nc):
            Gas_optics<TF>(band_lims_wavenum, band2gpt)
{
    this->is_longwave = true;
    this->gas_names = gas_names;

    initialize_networks(file_name_weights, input_nc);
}

// Constructor of the shortwave variant.
template<typename TF>
Gas_optics_nn<TF>::Gas_optics_nn(
        const Array<std::string,1>& gas_names,
        const Array<int,2>& band2gpt,
        const Array<TF,2>& band_lims_wavenum,
        const Array<TF,1>& solar_source_quiet,
        const Array<TF,1>& solar_source_facular,
        const Array<TF,1>& solar_source_sunspot,
        const TF tsi_default,
        const TF mg_default,
        const TF sb_default,
        const std::string& file_name_weights,
        Netcdf_file& input_nc):
            Gas_optics<TF>(band_lims_wavenum, band2gpt)
{
    this->is_longwave = false;
    this->gas_names = gas_names;
    this->solar_source_quiet = solar_source_quiet;
    this->solar_source_facular = solar_source_facular;
    this->solar_source_sunspot = solar_source_sunspot;
    this->solar_source.set_dims(solar_source_quiet.get_dims());

    set_solar_variability(mg_default, sb_default);
    initialize_networks(file_name_weights, input_nc);
}

// Gas optics solver longwave variant.
template<typename TF>
void Gas_optics_nn<TF>::gas_optics(
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Array<TF,1>& tsfc,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Source_func_lw<TF>& sources,
        const Array<TF,2>& col_dry,
        const Array<TF,2>& tlev) const
{
    const int ncol = play.dim(1);
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();

    compute_tau_sources_nn(
            this->tlw_network, this->plk_network,
            ncol, nlay, ngpt, nband, this->idx_tropo,
            play.ptr(), plev.ptr(), 
            tlay.ptr(), tlev.ptr(),
            gas_desc, sources, 
            optical_props,
            this->lower_atm, this->upper_atm);

    //fill surface sources  
    lay2sfc_factor(tlay,tsfc,sources,ncol,nlay,nband);
}

// Gas optics solver shortwave variant.
//template<typename TF>
template<typename TF>
void Gas_optics_nn<TF>::gas_optics(
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Array<TF,2>& toa_src,
        const Array<TF,2>& col_dry) const
{   
    const int ncol = play.dim(1);
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();

    compute_tau_ssa_nn(
            this->ssa_network, this->tsw_network,
            ncol, nlay, ngpt, nband, this->idx_tropo,
            play.ptr(), plev.ptr(), tlay.ptr(), 
            gas_desc, optical_props,
            this->lower_atm, this->upper_atm);

    // External source function is constant.
    for (int igpt=1; igpt<=ngpt; ++igpt)
        for (int icol=1; icol<=ncol; ++icol)
            toa_src({icol, igpt}) = this->solar_source({igpt});
}

template<typename TF>
void Gas_optics_nn<TF>::set_solar_variability(
        const TF mg_index, const TF sb_index)
{
    constexpr TF a_offset = TF(0.1495954);
    constexpr TF b_offset = TF(0.00066696);

    for (int igpt=1; igpt<=this->solar_source_quiet.dim(1); ++igpt)
    {
        this->solar_source({igpt}) = this->solar_source_quiet({igpt})
                + (mg_index - a_offset) * this->solar_source_facular({igpt})
                + (sb_index - b_offset) * this->solar_source_sunspot({igpt});
    }
}

template<typename TF>
TF Gas_optics_nn<TF>::get_tsi() const
{
    const int n_gpt = this->get_ngpt();

    TF tsi = 0.;
    for (int igpt=1; igpt<=n_gpt; ++igpt)
        tsi += this->solar_source({igpt});

    return tsi;
}

template<typename TF>
void Gas_optics_nn<TF>::initialize_networks(
        const std::string& wgth_file,
        Netcdf_file& input_nc)
{
    const int n_lay = input_nc.get_dimension_size("lay");
    const int n_col = input_nc.get_dimension_size("col");

    Array<TF,2> p_lay(input_nc.get_variable<TF>("p_lay", {n_lay, n_col}), {n_col, n_lay});

    int idx_tropo = 0;
    for (int i=1; i<=n_lay; i++)
    {
        if (p_lay({1,i}) > 9948.431564193395)
            idx_tropo += 1;
    }

    this->lower_atm = (idx_tropo>0);
    this->upper_atm = (idx_tropo<n_lay);
    this->idx_tropo = idx_tropo;

    Netcdf_file nc_wgth(wgth_file, Netcdf_mode::Read);

    const int n_layers = nc_wgth.get_dimension_size("nlayers");
    const int n_layer1 = nc_wgth.get_dimension_size("nlayer1");
    const int n_layer2 = nc_wgth.get_dimension_size("nlayer2");
    const int n_layer3 = nc_wgth.get_dimension_size("nlayer3");
    const int n_out_sw = nc_wgth.get_dimension_size("nout_sw");
    const int n_out_lw = nc_wgth.get_dimension_size("nout_lw");
    const int n_o3 = nc_wgth.get_dimension_size("ngases");

    const int n_in = 3 + n_o3;
    const int n_gpt = this->get_ngpt();

    if (n_gpt == n_out_lw)
    {
        const int n_out_pk = n_out_lw * 3;
        const int n_in_pk  = n_in + 2;
        Netcdf_group tlwnc = nc_wgth.get_group("TLW");
        this->tlw_network = Network(tlwnc,
                                    n_layers, n_layer1, n_layer2, n_layer3,
                                    n_out_lw, n_in);

        Netcdf_group plknc = nc_wgth.get_group("Planck");
        this->plk_network = Network(plknc,
                                    n_layers, n_layer1, n_layer2, n_layer3,
                                    n_out_pk, n_in_pk);
    }
    else if (n_gpt == n_out_sw)
    {
        Netcdf_group tswnc = nc_wgth.get_group("TSW");
        this->tsw_network = Network(tswnc,
                                   n_layers, n_layer1, n_layer2, n_layer3,
                                   n_out_sw, n_in);

        Netcdf_group ssanc = nc_wgth.get_group("SSA");
        this->ssa_network = Network(ssanc,
                                    n_layers, n_layer1, n_layer2, n_layer3,
                                    n_out_sw, n_in);
    }
    else
    {
        throw std::runtime_error("Network output size does not match the number of shortwave or longwave g-points.");
    }

    this->n_layers = n_layers;
    this->n_layer1 = n_layer1;
    this->n_layer2 = n_layer2;
    this->n_layer3 = n_layer3;
    this->n_o3 = n_o3;
}

template<typename TF>
void Gas_optics_nn<TF>::lay2sfc_factor(
        const Array<TF,2>& tlay,
        const Array<TF,1>& tsfc,
        Source_func_lw<TF>& sources,
        const int ncol,
        const int nlay,
        const int nband) const
{
    Array<TF,1> sfc_factor({nband});

    const Array<TF,3>& src_layer = sources.get_lay_source();
    Array<TF,2>& src_sfc = sources.get_sfc_source();

    for (int icol=1; icol<=ncol; ++icol)
    {
        //numbers are fitting from longwave coefficients file
        sfc_factor({1})  = pow((TF(0.0131757912608200)*tsfc({icol})-TF(1.)) / (TF(0.01317579126081997)*tlay({icol,1})-TF(1.)), TF(1.1209724347746475));
        sfc_factor({2})  = pow((TF(0.0092778215915162)*tsfc({icol})-TF(1.)) / (TF(0.00927782159151618)*tlay({icol,1})-TF(1.)), TF(1.4149505728750649));
        sfc_factor({3})  = pow((TF(0.0081221064580734)*tsfc({icol})-TF(1.)) / (TF(0.00812210645807336)*tlay({icol,1})-TF(1.)), TF(1.7153859296550862));
        sfc_factor({4})  = pow((TF(0.0078298195508505)*tsfc({icol})-TF(1.)) / (TF(0.00782981955085045)*tlay({icol,1})-TF(1.)), TF(1.9129486781120648));
        sfc_factor({5})  = pow((TF(0.0076928950299874)*tsfc({icol})-TF(1.)) / (TF(0.00769289502998736)*tlay({icol,1})-TF(1.)), TF(2.121924616912191));
        sfc_factor({6})  = pow((TF(0.0075653865084563)*tsfc({icol})-TF(1.)) / (TF(0.00756538650845630)*tlay({icol,1})-TF(1.)), TF(2.4434431185689567));
        sfc_factor({7})  = pow((TF(0.0074522945371839)*tsfc({icol})-TF(1.)) / (TF(0.00745229453718388)*tlay({icol,1})-TF(1.)), TF(2.7504289450500714));
        sfc_factor({8})  = pow((TF(0.0074105017545267)*tsfc({icol})-TF(1.)) / (TF(0.00741050175452665)*tlay({icol,1})-TF(1.)), TF(2.9950297268205865));
        sfc_factor({9})  = pow((TF(0.0074119101719575)*tsfc({icol})-TF(1.)) / (TF(0.00741191017195750)*tlay({icol,1})-TF(1.)), TF(3.3798218227597565));
        sfc_factor({10}) = pow((TF(0.0073401394763094)*tsfc({icol})-TF(1.)) / (TF(0.00734013947630943)*tlay({icol,1})-TF(1.)), TF(3.760811429547177));
        sfc_factor({11}) = pow((TF(0.0074075710256119)*tsfc({icol})-TF(1.)) / (TF(0.00740757102561185)*tlay({icol,1})-TF(1.)), TF(4.267112286396149));
        sfc_factor({12}) = pow((TF(0.0073542571820469)*tsfc({icol})-TF(1.)) / (TF(0.00735425718204687)*tlay({icol,1})-TF(1.)), TF(5.037348344205931));
        sfc_factor({13}) = pow((TF(0.0073059753467312)*tsfc({icol})-TF(1.)) / (TF(0.00730597534673117)*tlay({icol,1})-TF(1.)), TF(5.629568565488524));
        sfc_factor({14}) = pow((TF(0.0072946170050561)*tsfc({icol})-TF(1.)) / (TF(0.00729461700505611)*tlay({icol,1})-TF(1.)), TF(6.032163655628699));
        sfc_factor({15}) = pow((TF(0.0073266552903883)*tsfc({icol})-TF(1.)) / (TF(0.00732665529038828)*tlay({icol,1})-TF(1.)), TF(6.566161469007115));
        sfc_factor({16}) = pow((TF(0.0074129528692143)*tsfc({icol})-TF(1.)) / (TF(0.00741295286921425)*tlay({icol,1})-TF(1.)), TF(7.579678774748928));

        for (int iband=1; iband<=nband; ++iband)
            for (int igpt=1; igpt<=16; ++igpt)
            {
                const int idxgpt = igpt + 16 * (iband-1);
                src_sfc({icol,idxgpt}) = sfc_factor({iband}) * src_layer({icol,1,idxgpt});
            }           
    }                       
}
 
//Neural Network optical property function for shortwave
//Currently only implemented for atmospheric profilfes ordered bottom-first
template<typename TF>
void Gas_optics_nn<TF>::compute_tau_ssa_nn(
        const Network& nw_ssa,
        const Network& nw_tsw,
        const int ncol, const int nlay, const int ngpt, const int nband, const int idx_tropo,
        const TF* restrict const play,
        const TF* restrict const plev,
        const TF* restrict const tlay,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        const bool lower_atm, const bool upper_atm) const
{
    const int nlay_in = 3 + this->n_o3; //minimum input: h2o,T,P
    TF* tau = optical_props->get_tau().ptr();
    TF* ssa = optical_props->get_ssa().ptr();
    
    int startidx = 0;

    float dp[ncol*nlay];
    for (int ilay=0; ilay<nlay; ++ilay)
        for (int icol=0; icol<ncol; ++icol)
        {
            const int dpidx = icol+ilay*ncol;
            dp[dpidx] = abs(plev[dpidx]-plev[dpidx+ncol]);
        }

    //get gas concentrations
    const TF* h2o = gas_desc.get_vmr(this->gas_names({1})).ptr();
    const TF* o3  = gas_desc.get_vmr(this->gas_names({3})).ptr();

    std::vector<float> input;
    std::vector<float> output_tau;
    std::vector<float> output_ssa;

    const int nbatch_lower = ncol*idx_tropo;
    const int nbatch_upper = ncol*(nlay-idx_tropo);
    const int nbatch_max = std::max(nbatch_lower, nbatch_upper);

    input.resize(nbatch_max*nlay_in);

    output_tau.resize(nbatch_max*ngpt);
    output_ssa.resize(nbatch_max*ngpt);

    if (lower_atm) // Lower atmosphere:
    {
        //fill input array
        for (int i=0; i<idx_tropo; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val = logarithm(h2o[j+i*ncol]);
                const int idx = j+i*ncol;
                input[idx] = val;
            }
        
        if (this->n_o3 == 1)
        {
            startidx += ncol * idx_tropo;
            for (int i=0; i<idx_tropo; ++i)
                for (int j=0; j<ncol; ++j)
                {
                    const float val = logarithm(o3[j+i*ncol]);
                    const int idx   = startidx + j+i*ncol;
                    input[idx] = val;
                }
        }
        
        startidx += ncol * idx_tropo;
        for (int i=0; i<idx_tropo; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val = logarithm(play[j+i*ncol]);
                const int idx   = startidx + j+i*ncol;
                input[idx] = val;
            }

        startidx += ncol * idx_tropo;
        for (int i=0; i<idx_tropo; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val = tlay[j+i*ncol];
                const int idx   = startidx + j+i*ncol;
                input[idx] = val;
            }

        nw_tsw.inference(input.data(), output_tau.data(), nbatch_lower, 1,1,1, this->n_layers, this->n_layer1, this->n_layer2, this->n_layer3); //lower atmosphere, exp(output), normalize input
        nw_ssa.inference(input.data(), output_ssa.data(), nbatch_lower, 1,0,0, this->n_layers, this->n_layer1, this->n_layer2, this->n_layer3); //lower atmosphere, output, input already normalized);
   
        copy_arrays_ssa(output_ssa.data(), ssa, ncol, 0, idx_tropo, ngpt, nlay);
        copy_arrays_tau(output_tau.data(), dp, tau, ncol, 0, idx_tropo, ngpt, nlay);
    }
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++     
    if (upper_atm) //// Upper atmosphere:
    {
        // Fill input array.
        startidx = 0;
        for (int i=idx_tropo; i<nlay; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val = logarithm(h2o[j+i*ncol]);
                const int idx = j+(i-idx_tropo)*ncol;
                input[idx] = val;
            }

        if (this->n_o3 == 1)
        {
            startidx += ncol*(nlay-idx_tropo);
            for (int i=idx_tropo; i<nlay; ++i)
                for (int j=0; j<ncol; ++j)
                {
                    const float val = logarithm(o3[j+i*ncol]);
                    const int idx   = startidx + j+(i-idx_tropo)*ncol;
                    input[idx] = val;
                }
        } 
        
        startidx += ncol*(nlay-idx_tropo);
        for (int i=idx_tropo; i<nlay; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val = logarithm(play[j+i*ncol]);
                const int idx   = startidx + j+(i-idx_tropo)*ncol;
                input[idx] = val;
            }

        startidx += ncol*(nlay-idx_tropo);
        for (int i=idx_tropo; i<nlay; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val = tlay[j+i*ncol];
                const int idx   = startidx + j+(i-idx_tropo)*ncol;
                input[idx] = val;
            }

        nw_tsw.inference(input.data(), output_tau.data(), nbatch_upper, 0,1,1, this->n_layers, this->n_layer1, this->n_layer2, this->n_layer3); //upper atmosphere, exp(output), normalize input
        nw_ssa.inference(input.data(), output_ssa.data(), nbatch_upper, 0,0,0, this->n_layers, this->n_layer1, this->n_layer2, this->n_layer3); //upper atmosphere, output, input already normalized

        copy_arrays_ssa(output_ssa.data(), ssa, ncol, idx_tropo, nlay, ngpt, nlay);
        copy_arrays_tau(output_tau.data(), dp, tau, ncol, idx_tropo, nlay, ngpt, nlay);
    }
}

// Neural Network optical property function for longwave.
// Currently only implemented for atmospheric profiles ordered bottom-first.
template<typename TF>
void Gas_optics_nn<TF>::compute_tau_sources_nn(
        const Network& nw_tlw,
        const Network& nw_plk,
        const int ncol, const int nlay, const int ngpt, const int nband, const int idx_tropo,
        const TF* restrict const play,
        const TF* restrict const plev,
        const TF* restrict const tlay,
        const TF* restrict const tlev,
        const Gas_concs<TF>& gas_desc,
        Source_func_lw<TF>& sources,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        const bool lower_atm, const bool upper_atm) const
{
    const int nlay_in = 3 + this->n_o3; //minimum input: h2o,T,P

    TF* tau = optical_props->get_tau().ptr();
    TF* src_layer = sources.get_lay_source().ptr();
    TF* src_lvinc = sources.get_lev_source_inc().ptr();
    TF* src_lvdec = sources.get_lev_source_dec().ptr();

    int startidx = 0;
    int startidx2 =0;

    float dp[ncol*nlay];
    for (int ilay=0; ilay<nlay; ++ilay)
        for (int icol=0; icol<ncol; ++icol)
        {
            const int dpidx = icol + ilay * ncol;
            dp[dpidx] = std::abs(plev[dpidx]-plev[dpidx+ncol]);
        }
    
    // Get gas concentrations.
    const TF* h2o = gas_desc.get_vmr(this->gas_names({1})).ptr();
    const TF* o3  = gas_desc.get_vmr(this->gas_names({3})).ptr();

    std::vector<float> input_tau;
    std::vector<float> input_plk;
    std::vector<float> output_tau;
    std::vector<float> output_plk;

    const int nbatch_lower = ncol*idx_tropo;
    const int nbatch_upper = ncol*(nlay-idx_tropo);
    const int nbatch_max = std::max(nbatch_lower, nbatch_upper);
    input_tau .resize(nbatch_max*nlay_in);
    input_plk .resize(nbatch_max*(nlay_in+2));
    output_tau.resize(nbatch_max*ngpt);
    output_plk.resize(nbatch_max*ngpt*3);

    if (lower_atm) //// Lower atmosphere:
    {
        //fill input arrays
        for (int i=0; i<idx_tropo; ++i)
            for (int j=0; j<ncol; ++j)
            { 
                const float val = logarithm(h2o[j+i*ncol]);
                const int idx = j + i*ncol;
                input_tau[idx] = val;
                input_plk[idx] = val;
            }

        if (this->n_o3 == 1)
        {
            startidx += ncol * idx_tropo;
            for (int i=0; i<idx_tropo; ++i)
                for (int j=0; j<ncol; ++j)
                {
                    const float val = logarithm(o3[j+i*ncol]);
                    const int idx = startidx + j + i*ncol;
                    input_tau[idx] = val;
                    input_plk[idx] = val;
                }
        }

        startidx += ncol * idx_tropo;
        for (int i=0; i<idx_tropo; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val = logarithm(play[j+i*ncol]);
                const int idx = startidx + j + i*ncol;
                input_tau[idx] = val;
                input_plk[idx] = val;
            }

        startidx += ncol * idx_tropo;
        for (int i=0; i<idx_tropo; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val = tlay[j+i*ncol];
                const int idx = startidx + j + i*ncol;
                input_tau[idx] = val;
                input_plk[idx] = val;
            }

        startidx += ncol * idx_tropo;
        startidx2 = startidx + ncol * idx_tropo;
        for (int i=0; i<idx_tropo; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val1 = tlev[j+i*ncol];
                const float val2 = tlev[j+(i+1)*ncol];
                const int idx1 = startidx + j+i*ncol;
                const int idx2 = startidx2 + j+i*ncol;
                input_plk[idx1] = val1;
                input_plk[idx2] = val2;
            }

        nw_tlw.inference(input_tau.data(), output_tau.data(), nbatch_lower, 1,1,1, this->n_layers, this->n_layer1, this->n_layer2, this->n_layer3); //lower atmosphere, exp(output), normalize input
        nw_plk.inference(input_plk.data(), output_plk.data(), nbatch_lower, 1,1,1, this->n_layers, this->n_layer1, this->n_layer2, this->n_layer3); //lower atmosphere, exp(output), normalize input

        copy_arrays_tau(output_tau.data(), dp, tau, ncol, 0, idx_tropo, ngpt, nlay);
        copy_arrays_plk(output_plk.data(), src_layer, src_lvinc, src_lvdec,ncol, 0, idx_tropo, ngpt, nlay);
        // We swap lvdec and lvinc with respect to neural network training data, which was generated with a top-bottom ordering.
    }
    if (upper_atm) //// Upper atmosphere:
    {
        // Fill input array.
        startidx = 0;
        for (int i=idx_tropo;i< nlay; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = logarithm(h2o[j+i*ncol]);
                const int idx = j+(i-idx_tropo)*ncol;
                input_tau[idx] = val;
                input_plk[idx] = val;
            }

        if (this->n_o3 == 1)
        {
            startidx += ncol*(nlay-idx_tropo);
            for (int i=idx_tropo; i<nlay; ++i)
                for (int j=0; j<ncol; ++j)
                {
                    const float val = logarithm(o3[j+i*ncol]);
                    const int idx = startidx + j+(i-idx_tropo)*ncol;
                    input_tau[idx] = val;
                    input_plk[idx] = val;
                }
        }

        startidx += ncol*(nlay-idx_tropo);
        for (int i=idx_tropo; i<nlay; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val = logarithm(play[j+i*ncol]);
                const int idx = startidx + j+(i-idx_tropo)*ncol;
                input_tau[idx] = val;
                input_plk[idx] = val;
            }

        startidx += ncol*(nlay-idx_tropo);
        for (int i=idx_tropo; i<nlay; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val = tlay[j+i*ncol];
                const int idx = startidx + j+(i-idx_tropo)*ncol;
                input_tau[idx] = val;
                input_plk[idx] = val;
            }
        
        startidx += ncol*(nlay-idx_tropo);
        startidx2 = startidx+ncol*(nlay-idx_tropo);

        for (int i=idx_tropo; i<nlay; ++i)
            for (int j=0; j<ncol; ++j)
            {
                const float val1 = tlev[j+i*ncol];
                const float val2 = tlev[j+(i+1)*ncol];
                const int idx1 = startidx  + j+(i-idx_tropo)*ncol;
                const int idx2 = startidx2 + j+(i-idx_tropo)*ncol;
                input_plk[idx1] = val1;
                input_plk[idx2] = val2;
            }

        nw_tlw.inference(input_tau.data(), output_tau.data(), nbatch_upper, 0,1,1, this->n_layers, this->n_layer1, this->n_layer2, this->n_layer3); //upper atmosphere, exp(output), normalize input
        nw_plk.inference(input_plk.data(), output_plk.data(), nbatch_upper, 0,1,1, this->n_layers, this->n_layer1, this->n_layer2, this->n_layer3); //upper atmosphere, exp(output), normalize input
 
        copy_arrays_tau(output_tau.data(), dp, tau, ncol, idx_tropo, nlay, ngpt, nlay);
        copy_arrays_plk(output_plk.data(), src_layer, src_lvinc, src_lvdec, ncol, idx_tropo, nlay, ngpt, nlay);
        // We swap lvdec and lvinc with respect to neural network training data, which was generated with a top-bottom ordering.
    }
}

#ifdef FLOAT_SINGLE_RRTMGP
template class Gas_optics_nn<float>;
#else
template class Gas_optics_nn<double>;
#endif

