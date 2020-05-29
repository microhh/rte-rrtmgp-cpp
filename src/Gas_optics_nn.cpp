#include <cmath>
#include <numeric>
#include <boost/algorithm/string.hpp>
#include "Gas_concs.h"
#include "Netcdf_interface.h"
#include "Gas_opticsNN.h"
#include "Array.h"
#include "Optical_props.h"
#include "Source_functions.h"
#include "rrtmgp_kernels.h"
#include <time.h>
#include <sys/time.h>

#define restrict __restrict__

double get_wall_time2()
{
    struct timeval time;
    if (gettimeofday(&time,NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

namespace
{
    double starttime,endtime;
    double starttimeX,endtimeX;
    inline float mylog(float x)
    {
        x = sqrt(x);x = sqrt(x);
        x = sqrt(x);x = sqrt(x);
        x = (x-1.0f) * 16.0f;
        return x;
    }
 
    void copy_arrays_tau(
                 const float* restrict const data_in,
                 const float* restrict const data_dp,
                 double* restrict const data_out,
                 const int N1,  const int N2a, 
                 const int N2b, const int N3,
                 const int nlay)
    {
        const float* dp_temp = &data_dp[N1*N2a];
        const int Nup = N2b-N2a;
        for (int i = 0; i < N3; ++i)
        {
            const int outidx = i*nlay*N1+N2a*N1;
            const float* in_temp = &data_in[i*Nup*N1];
            double* out_temp = &data_out[outidx];
            #pragma ivdep            
            for (int j = 0; j < N1*Nup; ++j)
            {
                out_temp[j] = in_temp[j] * dp_temp[j]; 
            }             
        }
    }

    void copy_arrays_ssa(
                 const float* restrict const data_in,
                 double* restrict const data_out,
                 const int N1,  const int N2a, 
                 const int N2b, const int N3,
                 const int nlay)
    {
        const int Nup = N2b-N2a;
        for (int i = 0; i < N3; ++i)
        {
            const int outidx = i*nlay*N1+N2a*N1;
            const float* in_temp = &data_in[i*Nup*N1];
            double* out_temp = &data_out[outidx];
            #pragma ivdep            
            for (int j = 0; j < N1*Nup; ++j)
            {
                out_temp[j] = in_temp[j]; 
            }             
        }
    }

    void copy_arrays_plk(
                 const float* restrict const data_in,
                 double* restrict const data_out1,
                 double* restrict const data_out2,
                 double* restrict const data_out3,
                 const int N1,  const int N2a, 
                 const int N2b, const int N3,
                 const int nlay)
    {
        const int Nup = N2b-N2a;
        for (int i = 0; i < N3; ++i)
        {
            const int outidx = i*nlay*N1+N2a*N1;
            const float* in_temp1 = &data_in[i*Nup*N1];
            const float* in_temp2 = &data_in[(i+N3)*Nup*N1];
            const float* in_temp3 = &data_in[(i+N3+N3)*Nup*N1];
            double* out_temp1= &data_out1[outidx];
            double* out_temp2= &data_out2[outidx];
            double* out_temp3= &data_out3[outidx];
            #pragma ivdep           
            for (int j = 0; j < N1*Nup; ++j)
            {
                out_temp1[j] = in_temp1[j];
                out_temp2[j] = in_temp2[j];
                out_temp3[j] = in_temp3[j];
            
            }
        }
    }

}
       
//     // Constructor of longwave variant.
//template<typename TF>
template<typename TF,int Nlayer,int N_gas,int N_lay1,int N_lay2,int N_lay3>
Gas_opticsNN<TF,Nlayer,N_gas,N_lay1,N_lay2,N_lay3>::Gas_opticsNN(
        const Array<std::string,1>& gas_names,
        const Array<int,2>& band2gpt,
        const Array<TF,2>& band_lims_wavenum):
            Optical_props<TF>(band_lims_wavenum, band2gpt)
{
    this->is_longwave = true;
    this->gas_names = gas_names;
}

// Constructor of the shortwave variant.
template<typename TF,int Nlayer,int N_gas,int N_lay1,int N_lay2,int N_lay3>
Gas_opticsNN<TF,Nlayer,N_gas,N_lay1,N_lay2,N_lay3>::Gas_opticsNN(
        const Array<std::string,1>& gas_names,
        const Array<int,2>& band2gpt,
        const Array<TF,2>& band_lims_wavenum,
        const Array<TF,1>& solar_src,
        const bool do_taussa):
            Optical_props<TF>(band_lims_wavenum, band2gpt),
            solar_src(solar_src)
{ 
   
    this->is_longwave = false;   
    this->do_taussa = do_taussa;
    this->gas_names = gas_names;
}

// Gas optics solver longwave variant.
//template<typename TF>
template<typename TF,int Nlayer,int N_gas,int N_lay1,int N_lay2,int N_lay3>
void Gas_opticsNN<TF,Nlayer,N_gas,N_lay1,N_lay2,N_lay3>::gas_optics(
        Network<Nlayer,N_lay1,N_lay2,N_lay3>& TLW,
        Network<Nlayer,N_lay1,N_lay2,N_lay3>& PLK,
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Array<TF,1>& tsfc,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Source_func_lw<TF>& sources,
        const Array<TF,2>& tlev,
        const int idx_tropo, 
        const bool lower_atm, const bool upper_atm) const
{
    const int ncol = play.dim(1);
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();

    compute_tau_sources_NN(TLW, PLK,
            ncol, nlay, ngpt, nband, idx_tropo, 
            play.ptr(), plev.ptr(), 
            tlay.ptr(), tlev.ptr(),
            gas_desc, sources, 
            optical_props,
            lower_atm, upper_atm);   

    //fill surface sources  
    lay2sfc_factor(tlay,tsfc,sources,ncol,nlay,nband);   
}

// Gas optics solver shortwave variant.
//template<typename TF>
template<typename TF,int Nlayer,int N_gas,int N_lay1,int N_lay2,int N_lay3>
void Gas_opticsNN<TF,Nlayer,N_gas,N_lay1,N_lay2,N_lay3>::gas_optics(
        Network<Nlayer,N_lay1,N_lay2,N_lay3>& SSA, 
        Network<Nlayer,N_lay1,N_lay2,N_lay3>& TSW, 
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Array<TF,2>& toa_src,
        const int idx_tropo, 
        const bool lower_atm, const bool upper_atm) const
{   
    const int ncol = play.dim(1);
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();
    compute_tau_ssa_NN(
            SSA,TSW,
            ncol, nlay, ngpt, nband, idx_tropo, 
            play.ptr(), plev.ptr(), tlay.ptr(), 
            gas_desc, optical_props,
            lower_atm, upper_atm);

    // External source function is constant.
    for (int igpt=1; igpt<=ngpt; ++igpt)
        for (int icol=1; icol<=ncol; ++icol)
            toa_src({icol, igpt}) = this->solar_src({igpt});
}

template<typename TF,int Nlayer,int N_gas,int N_lay1,int N_lay2,int N_lay3>
void Gas_opticsNN<TF,Nlayer,N_gas,N_lay1,N_lay2,N_lay3>::lay2sfc_factor(
        const Array<TF,2>& tlay,
        const Array<TF,1>& tsfc,
        Source_func_lw<TF>& sources,
        const int& ncol,
        const int& nlay,
        const int& nband) const
{
    Array<TF,1> sfc_factor({nband});
    Array<TF,3>& src_layer = sources.get_lay_source();
    Array<TF,2>& src_sfc   = sources.get_sfc_source();
    for (int icol=1; icol<=ncol; ++icol)
    {
        const float tempfrac = tsfc({icol})/tlay({icol,1});
        sfc_factor({1}) = pow((0.013175791260819966*tsfc({icol})-1) / (0.013175791260819966*tlay({icol,1})-1),1.1209724347746475);
        sfc_factor({2}) = pow((0.009277821591516187*tsfc({icol})-1) / (0.009277821591516187*tlay({icol,1})-1),1.4149505728750649);
        sfc_factor({3}) = pow((0.008122106458073356*tsfc({icol})-1) / (0.008122106458073356*tlay({icol,1})-1),1.7153859296550862);
        sfc_factor({4}) = pow((0.00782981955085045*tsfc({icol})-1) / (0.00782981955085045*tlay({icol,1})-1),1.9129486781120648);
        sfc_factor({5}) = pow((0.007692895029987358*tsfc({icol})-1) / (0.007692895029987358*tlay({icol,1})-1),2.121924616912191);
        sfc_factor({6}) = pow((0.0075653865084563034*tsfc({icol})-1) / (0.0075653865084563034*tlay({icol,1})-1),2.4434431185689567);
        sfc_factor({7}) = pow((0.007452294537183882*tsfc({icol})-1) / (0.007452294537183882*tlay({icol,1})-1),2.7504289450500714);
        sfc_factor({8}) = pow((0.007410501754526651*tsfc({icol})-1) / (0.007410501754526651*tlay({icol,1})-1),2.9950297268205865);
        sfc_factor({9}) = pow((0.007411910171957498*tsfc({icol})-1) / (0.007411910171957498*tlay({icol,1})-1),3.3798218227597565);
        sfc_factor({10}) = pow((0.00734013947630943*tsfc({icol})-1) / (0.00734013947630943*tlay({icol,1})-1),3.760811429547177);
        sfc_factor({11}) = pow((0.007407571025611854*tsfc({icol})-1) / (0.007407571025611854*tlay({icol,1})-1),4.267112286396149);
        sfc_factor({12}) = pow((0.007354257182046865*tsfc({icol})-1) / (0.007354257182046865*tlay({icol,1})-1),5.037348344205931);
        sfc_factor({13}) = pow((0.007305975346731165*tsfc({icol})-1) / (0.007305975346731165*tlay({icol,1})-1),5.629568565488524);
        sfc_factor({14}) = pow((0.00729461700505611*tsfc({icol})-1) / (0.00729461700505611*tlay({icol,1})-1),6.032163655628699);
        sfc_factor({15}) = pow((0.007326655290388281*tsfc({icol})-1) / (0.007326655290388281*tlay({icol,1})-1),6.566161469007115);
        sfc_factor({16}) = pow((0.00741295286921425*tsfc({icol})-1) / (0.00741295286921425*tlay({icol,1})-1),7.579678774748928);
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
template<typename TF,int Nlayer,int N_gas,int N_lay1,int N_lay2,int N_lay3>
void Gas_opticsNN<TF,Nlayer,N_gas,N_lay1,N_lay2,N_lay3>::compute_tau_ssa_NN(
        Network<Nlayer,N_lay1,N_lay2,N_lay3>& SSA, 
        Network<Nlayer,N_lay1,N_lay2,N_lay3>& TSW,
        const int ncol, const int nlay, const int ngpt, const int nband, const int idx_tropo,
        const double* restrict const play,
        const double* restrict const plev,
        const double* restrict const tlay,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        const bool lower_atm, const bool upper_atm) const
{
    constexpr int N_layI = N_gas + 3; //minimum input: h2o,T,P
    double* tau = optical_props->get_tau().ptr();
    double* ssa = optical_props->get_ssa().ptr();
    
    float nul = 0.;
    float een = 1.;
    int startidx = 0;

    float dp[ncol * nlay];
    for (int ilay=0; ilay<nlay; ++ilay)
        for (int icol=0; icol<ncol; ++icol)
        {
            const int dpidx = icol + ilay * ncol;
            dp[dpidx] = abs(plev[dpidx]-plev[dpidx+ncol]);
        }

    //get gas concentrations
    const double* h2o = gas_desc.get_vmr(this->gas_names({1})).ptr();
    const double* o3  = gas_desc.get_vmr(this->gas_names({3})).ptr();
    
    
    if (lower_atm) //// Lower atmosphere:
    {
        //fill input array  
        float input_lower[ncol*idx_tropo*N_layI];
        float output_lower_tau[ncol*idx_tropo*ngpt];
        float output_lower_ssa[ncol*idx_tropo*ngpt];

        for (int i = 0; i < idx_tropo; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = mylog(h2o[j+i*ncol]);
                const int idx = j+i*ncol;
                input_lower[idx] = val;
            }
        
        if constexpr (N_gas == 1)
        {
            startidx += ncol * idx_tropo;
            for (int i = 0; i < idx_tropo; ++i)
                for (int j = 0; j < ncol; ++j)
                {
                    const float val = mylog(o3[j+i*ncol]);
                    const int idx   = startidx + j+i*ncol;
                    input_lower[idx] = val;
                }
        }
        
        startidx += ncol * idx_tropo;
        for (int i = 0; i < idx_tropo; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = mylog(play[j+i*ncol]);
                const int idx   = startidx + j+i*ncol;
                input_lower[idx] = val;
            }

        startidx += ncol * idx_tropo;
        for (int i = 0; i < idx_tropo; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = tlay[j+i*ncol];
                const int idx   = startidx + j+i*ncol;
                input_lower[idx] = val;
            }



        TSW.Inference(input_lower, output_lower_tau, 1,1,1); //lower atmosphere, exp(output), normalize input
        SSA.Inference(input_lower, output_lower_ssa, 1,0,0); //lower atmosphere, output, input already normalized);
   
        copy_arrays_ssa(output_lower_ssa,ssa,ncol,   0,idx_tropo,ngpt,nlay);    
        copy_arrays_tau(output_lower_tau,dp,tau,ncol,0,idx_tropo,ngpt,nlay); 
    }
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++     
    if (upper_atm) //// Upper atmosphere:
    {
        //fill input array
        float input_upper     [ncol*(nlay-idx_tropo)*N_layI];
        float output_upper_tau[ncol*(nlay-idx_tropo)*ngpt];
        float output_upper_ssa[ncol*(nlay-idx_tropo)*ngpt];

        startidx = 0;
        for (int i = idx_tropo; i < nlay; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = mylog(h2o[j+i*ncol]);
                const int idx = j+(i-idx_tropo)*ncol;
                input_upper[idx] = val;
            }

        if constexpr (N_gas == 1)
        {
            startidx += ncol*(nlay-idx_tropo);
            for (int i = idx_tropo; i < nlay; ++i)
                for (int j = 0; j < ncol; ++j)
                {
                    const float val = mylog(o3[j+i*ncol]);
                    const int idx   = startidx + j+(i-idx_tropo)*ncol;
                    input_upper[idx] = val;
                }
        } 
        
        startidx += ncol*(nlay-idx_tropo);
        for (int i = idx_tropo; i < nlay; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = mylog(play[j+i*ncol]);
                const int idx   = startidx + j+(i-idx_tropo)*ncol;
                input_upper[idx] = val;
            }

        startidx += ncol*(nlay-idx_tropo);
        for (int i = idx_tropo; i < nlay; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = tlay[j+i*ncol];
                const int idx   = startidx + j+(i-idx_tropo)*ncol;
                input_upper[idx] = val;
            }

        TSW.Inference(input_upper, output_upper_tau, 0,1,1); //upper atmosphere, exp(output), normalize input
        SSA.Inference(input_upper, output_upper_ssa, 0,0,0); //upper atmosphere, output, input already normalized 
        
        copy_arrays_ssa(output_upper_ssa,ssa,ncol,idx_tropo,   nlay,ngpt,nlay);
        copy_arrays_tau(output_upper_tau,dp,tau,ncol,idx_tropo,nlay,ngpt,nlay);
    }
}

//Neural Network optical property function for longwave
//Currently only implemented for atmospheric profilfes ordered bottom-first
template<typename TF,int Nlayer,int N_gas,int N_lay1,int N_lay2,int N_lay3>
void Gas_opticsNN<TF,Nlayer,N_gas,N_lay1,N_lay2,N_lay3>::compute_tau_sources_NN(
        Network<Nlayer,N_lay1,N_lay2,N_lay3>& TLW,
        Network<Nlayer,N_lay1,N_lay2,N_lay3>& PLK,
        const int ncol, const int nlay, const int ngpt, const int nband, const int idx_tropo,
        const double* restrict const play,
        const double* restrict const plev,
        const double* restrict const tlay,
        const double* restrict const tlev,
        const Gas_concs<TF>& gas_desc,
        Source_func_lw<TF>& sources,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        const bool lower_atm, const bool upper_atm) const
{
    constexpr int N_layI = N_gas + 3; //minimum input: h2o,T,P
    
    double* tau = optical_props->get_tau().ptr();
    double* src_layer = sources.get_lay_source().ptr();
    double* src_lvinc = sources.get_lev_source_inc().ptr();
    double* src_lvdec = sources.get_lev_source_dec().ptr();

    float nul = 0.;
    float een = 1.;
    int startidx = 0;
    int startidx2 =0;

    float dp[ncol * nlay];
    for (int ilay=0; ilay<nlay; ++ilay)
        for (int icol=0; icol<ncol; ++icol)
        {
            const int dpidx = icol + ilay * ncol;
            dp[dpidx] = abs(plev[dpidx]-plev[dpidx+ncol]);
        }
    
    //get gas concentrations
    const double* h2o = gas_desc.get_vmr(this->gas_names({1})).ptr();
    const double* o3  = gas_desc.get_vmr(this->gas_names({3})).ptr();

    if (lower_atm) //// Lower atmosphere:
    {
        //fill input array  
        float input_lower_tau[ncol*idx_tropo*N_layI];
        float input_lower_plk[ncol*idx_tropo*(N_layI+2)];
        float output_lower_tau[ncol*idx_tropo*ngpt];
        float output_lower_plk[ncol*idx_tropo*ngpt*3];

        for (int i = 0; i < idx_tropo; ++i)
            for (int j = 0; j < ncol; ++j)
            { 
                const float val = mylog(h2o[j+i*ncol]);
                const int idx   = j+i*ncol;
                input_lower_tau[idx] = val;
                input_lower_plk[idx] = val;
            }
        
        if constexpr (N_gas == 1)
        {
            startidx += ncol * idx_tropo;
            for (int i = 0; i < idx_tropo; ++i)
                for (int j = 0; j < ncol; ++j)
                {
                    const float val = mylog(o3[j+i*ncol]);
                    const int idx   = startidx + j+i*ncol;
                    input_lower_tau[idx] = val;
                    input_lower_plk[idx] = val;
                }
        }

        startidx += ncol * idx_tropo;
        for (int i = 0; i < idx_tropo; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = mylog(play[j+i*ncol]);
                const int idx   = startidx + j+i*ncol;
                input_lower_tau[idx] = val;
                input_lower_plk[idx] = val;
            }

        startidx += ncol * idx_tropo;
        for (int i = 0; i < idx_tropo; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = tlay[j+i*ncol];
                const int idx   = startidx + j+i*ncol;
                input_lower_tau[idx] = val;
                input_lower_plk[idx] = val;
            }

        startidx += ncol * idx_tropo;
        startidx2 = startidx + ncol * idx_tropo;
        for (int i = 0; i < idx_tropo; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val1 = tlev[j+(i+1)*ncol];
                const float val2 = tlev[j+i*ncol];
                const int idx1 = startidx + j+i*ncol;
                const int idx2 = startidx2 + j+i*ncol;
                input_lower_plk[idx1] = val1;
                input_lower_plk[idx2] = val2;
            }

        TLW.Inference(input_lower_tau, output_lower_tau, 1,1,1); //lower atmosphere, exp(output), normalize input
        PLK.Inference(input_lower_plk, output_lower_plk, 1,1,1); //lower atmosphere, exp(output), normalize input

        copy_arrays_tau(output_lower_tau,dp,tau,ncol,0,idx_tropo,ngpt,nlay);
        copy_arrays_plk(output_lower_plk,src_layer,src_lvdec,src_lvinc,ncol,0,idx_tropo,ngpt,nlay); 
        //We swap lvdec and lvinc with respect to neural network training data, which was generated with a top-bottom ordering
    }

    if (upper_atm) //// Upper atmosphere:
    {
        //fill input array
        float input_upper_tau [ncol*(nlay-idx_tropo)*N_layI];
        float input_upper_plk [ncol*(nlay-idx_tropo)*(N_layI+2)];
        float output_upper_tau[ncol*(nlay-idx_tropo)*ngpt];
        float output_upper_plk[ncol*(nlay-idx_tropo)*ngpt*3];   

        startidx = 0;
        for (int i = idx_tropo; i < nlay; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = mylog(h2o[j+i*ncol]);
                const int idx   = j+(i-idx_tropo)*ncol;
                input_upper_tau[idx] = val;
                input_upper_plk[idx] = val;
            }

        if constexpr (N_gas == 1)
        {
            startidx += ncol*(nlay-idx_tropo);
            for (int i = idx_tropo; i < nlay; ++i)
                for (int j = 0; j < ncol; ++j)
                {
                    const float val = mylog(o3[j+i*ncol]);
                    const int idx   = startidx + j+(i-idx_tropo)*ncol;
                    input_upper_tau[idx] = val;
                    input_upper_plk[idx] = val;
                }
        }

        startidx += ncol*(nlay-idx_tropo);
        for (int i = idx_tropo; i < nlay; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = mylog(play[j+i*ncol]);
                const int idx   = startidx + j+(i-idx_tropo)*ncol;
                input_upper_tau[idx] = val;
                input_upper_plk[idx] = val;
            }

        startidx += ncol*(nlay-idx_tropo);
        for (int i = idx_tropo; i < nlay; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val = tlay[j+i*ncol];
                const int idx   = startidx + j+(i-idx_tropo)*ncol;
                input_upper_tau[idx] = val;
                input_upper_plk[idx] = val;
            }
        
        startidx += ncol * (nlay-idx_tropo);
        startidx2 = startidx + ncol * (nlay-idx_tropo);
        for (int i = idx_tropo; i < nlay; ++i)
            for (int j = 0; j < ncol; ++j)
            {
                const float val1 = tlev[j+(i+1)*ncol];
                const float val2 = tlev[j+i*ncol];
                const int idx1 = startidx + j+(i-idx_tropo)*ncol;
                const int idx2 = startidx2 + j+(i-idx_tropo)*ncol;
                input_upper_plk[idx1] = val1;
                input_upper_plk[idx2] = val2;
            }

        TLW.Inference(input_upper_tau, output_upper_tau, 0,1,1); //upper atmosphere, exp(output), normalize input
        PLK.Inference(input_upper_plk, output_upper_plk, 0,1,1); //upper atmosphere, exp(output), normalize input
 
        copy_arrays_tau(output_upper_tau,dp,tau,ncol,idx_tropo,nlay,ngpt,nlay); 
        copy_arrays_plk(output_upper_plk,src_layer,src_lvdec,src_lvinc,ncol,idx_tropo,nlay,ngpt,nlay); 
        //We swap lvdec and lvinc with respect to neural network training data, which was generated with a top-bottom ordering
    }
}
#ifdef FLOAT_SINGLE_RRTMGP
template class Gas_opticsNN<float,Nlayer,Ngas,Nlay1,Nlay2,Nlay3>;
#else
template class Gas_opticsNN<double,Nlayer,Ngas,Nlay1,Nlay2,Nlay3>;
#endif

