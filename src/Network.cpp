//#include <math.h>
//#include <math.h>

#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "Netcdf_interface.h"
#include "Network.h"
//#include <mkl.h>
#include <cblas.h>
#include <time.h>
#include <sys/time.h>
#define restrict __restrict__
namespace
{
    inline float leaky_relu(const float a) {return std::max(0.2f*a,a);}

    inline void bias_and_activate(float* restrict output, const float* restrict bias, const int n_out, const int n_batch)
    {
        for (int i=0; i<n_out; ++i)
            #pragma ivdep
            for (int j=0; j<n_batch; ++j)
                output[j+i*n_batch] = leaky_relu(output[j+i*n_batch] + bias[i]);
    }

    void matmul_bias_act_blas(
            const int n_batch,
            const int n_row,
            const int n_col,
            const float* restrict weights,
            const float* restrict bias,
            float* restrict const layer_in,
            float* restrict const layer_out)
    {

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_row, n_batch, n_col, 1.f,
                    weights, n_col, layer_in, n_batch , 0.f, layer_out, n_batch);
        bias_and_activate(layer_out, bias, n_row, n_batch);
    }

    inline float exponential(float x)
    {
        x = 1.0f + x / 16.0f;
        x *= x;
        x *= x;
        x *= x;
        x *= x;
        return x;
    }

    void normalize_input(
            float* restrict const input, 
            const float* restrict const input_mean,
            const float* restrict const input_stdev,
            const int n_batch,
            const int n_lay_in)
    {
        for (int i=0; i<n_lay_in; ++i)
            #pragma ivdep
            for (int j=0; j<n_batch; ++j) {
                const int idxin = j + i * n_batch;
                input[idxin] = (input[idxin] - input_mean[i]) / input_stdev[i];
            }
    }

    template<int NLAYER, int NLAY1, int NLAY2, int NLAY3>
    void feedforward(
            float* restrict const input, 
            float* restrict const output,
            const float* restrict const layer1_wgth,
            const float* restrict const layer2_wgth,
            const float* restrict const layer3_wgth,
            const float* restrict const output_wgth,
            const float* restrict const layer1_bias,
            const float* restrict const layer2_bias,
            const float* restrict const layer3_bias,
            const float* restrict const output_bias,
            const float* restrict const input_mean,
            const float* restrict const input_stdev,
            const float* restrict const output_mean,
            const float* restrict const output_stdev,
            float* restrict const layer1,
            float* restrict const layer2,
            float* restrict const layer3,
            const int n_batch,
            const int n_lay_out,
            const int n_lay_in,
            const int do_exp,
            const int do_norm)
    {  
        if (do_norm) {normalize_input(input, input_mean, input_stdev, n_batch, n_lay_in);}

        if constexpr (NLAYER==0) //Linear regression
        {   
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_lay_out, n_batch, n_lay_in, 1.f,
                        output_wgth, n_lay_in, input, n_batch, 0.f, output, n_batch);
        }
        if constexpr (NLAYER==1)
        {   
            matmul_bias_act_blas(n_batch, NLAY1, n_lay_in, layer1_wgth, layer1_bias, input, layer1);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_lay_out, n_batch, NLAY1, 1.f,
                        output_wgth, NLAY1, layer1, n_batch, 0.f, output , n_batch);
        }
        if constexpr (NLAYER==2)
        {
            matmul_bias_act_blas(n_batch, NLAY1, n_lay_in, layer1_wgth, layer1_bias, input,  layer1);
            matmul_bias_act_blas(n_batch, NLAY2, NLAY1,    layer2_wgth, layer2_bias, layer1, layer2);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_lay_out, n_batch, NLAY2, 1.f,
                        output_wgth, NLAY2, layer2, n_batch, 0.f, output, n_batch);
        }
        if constexpr (NLAYER==3)
        {
            matmul_bias_act_blas(n_batch, NLAY1, n_lay_in, layer1_wgth, layer1_bias, input,  layer1);
            matmul_bias_act_blas(n_batch, NLAY2, NLAY1,    layer2_wgth, layer2_bias, layer1, layer2);
            matmul_bias_act_blas(n_batch, NLAY3, NLAY2,    layer3_wgth, layer3_bias, layer2, layer3);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_lay_out, n_batch, NLAY3, 1.f,
                        output_wgth, NLAY3, layer3, n_batch, 0.f, output, n_batch);
        }

        //output layer and denormalize
        if (do_exp==1)
        {
            for (int i=0; i<n_lay_out; ++i)
                #pragma ivdep
                for (int j=0; j<n_batch; ++j)
                {
                    const int layidx = j + i * n_batch;
                    output[layidx] = exponential((output[layidx] + output_bias[i]) * output_stdev[i] + output_mean[i]);
                }
        }
        else
        {
            for (int i=0; i<n_lay_out; ++i)
                #pragma ivdep
                for (int j=0; j<n_batch; ++j)
                {
                    const int layidx = j + i * n_batch;
                    output[layidx] = (output[layidx] +  output_bias[i]) * output_stdev[i] + output_mean[i] ;
                }
        }
    }
}

template<int NLAYER, int NLAY1, int NLAY2, int NLAY3>
void Network<NLAYER, NLAY1, NLAY2, NLAY3>::inference(
        float* inputs,
        float* outputs,
        const int lower_atmos,
        const int do_exp,
        const int do_inpnorm)
{
    if (lower_atmos == 1)
    {
        if constexpr (NLAYER>0) this->layer1.resize(NLAY1 * this->n_batch_lower);
        if constexpr (NLAYER>2) this->layer3.resize(NLAY3 * this->n_batch_lower);
        if constexpr (NLAYER>1) this->layer2.resize(NLAY2 * this->n_batch_lower);
        feedforward<NLAYER, NLAY1, NLAY2, NLAY3>(
            inputs,
            outputs,
            this->layer1_wgth_lower.data(),
            this->layer2_wgth_lower.data(),
            this->layer3_wgth_lower.data(),
            this->output_wgth_lower.data(),
            this->layer1_bias_lower.data(),
            this->layer2_bias_lower.data(),
            this->layer3_bias_lower.data(),
            this->output_bias_lower.data(),
            this->mean_input_lower.data(),
            this->stdev_input_lower.data(),
            this->mean_output_lower.data(),
            this->stdev_output_lower.data(),
            this->layer1.data(),
            this->layer2.data(),
            this->layer3.data(),
            this->n_batch_lower,
            this->n_lay_out,
            this->n_lay_in,
            do_exp,
            do_inpnorm);
    }
    else
    {
        if constexpr (NLAYER>0) this->layer1.resize(NLAY1 * this->n_batch_upper);
        if constexpr (NLAYER>1) this->layer2.resize(NLAY2 * this->n_batch_upper);
        if constexpr (NLAYER>2) this->layer3.resize(NLAY3 * this->n_batch_upper);
        feedforward<NLAYER, NLAY1, NLAY2, NLAY3>(
            inputs,
            outputs,
            this->layer1_wgth_upper.data(),
            this->layer2_wgth_upper.data(),
            this->layer3_wgth_upper.data(),
            this->output_wgth_upper.data(),
            this->layer1_bias_upper.data(),
            this->layer2_bias_upper.data(),
            this->layer3_bias_upper.data(),
            this->output_bias_upper.data(),
            this->mean_input_upper.data(),
            this->stdev_input_upper.data(),
            this->mean_output_upper.data(),
            this->stdev_output_upper.data(),
            this->layer1.data(),
            this->layer2.data(),
            this->layer3.data(),
            this->n_batch_upper,
            this->n_lay_out,
            this->n_lay_in,
            do_exp,
            do_inpnorm);
    }

}

template<int NLAYER, int NLAY1, int NLAY2, int NLAY3>
Network<NLAYER, NLAY1, NLAY2, NLAY3>::Network(const int n_batch_lower,
                                              const int n_batch_upper,
                                              Netcdf_group& grp,
                                              const int n_lay_out,
                                              const int n_lay_in)
{
    this->n_batch_lower = n_batch_lower;
    this->n_batch_upper = n_batch_upper;
    this->n_lay_out = n_lay_out;
    this->n_lay_in  = n_lay_in;

    if constexpr (NLAYER==0)
    {
        this->output_bias_lower = grp.get_variable<float>("bias1_lower",{n_lay_out});
        this->output_wgth_lower = grp.get_variable<float>("wgth1_lower",{n_lay_out, n_lay_in});
        this->output_bias_upper = grp.get_variable<float>("bias1_upper",{n_lay_out});
        this->output_wgth_upper = grp.get_variable<float>("wgth1_upper",{n_lay_out, n_lay_in});
    }
    else if constexpr (NLAYER == 1)
    {
        this->layer1_bias_lower = grp.get_variable<float>("bias1_lower",{NLAY1});
        this->output_bias_lower = grp.get_variable<float>("bias2_lower",{n_lay_out});
        this->layer1_wgth_lower = grp.get_variable<float>("wgth1_lower",{NLAY1,     n_lay_in});
        this->output_wgth_lower = grp.get_variable<float>("wgth2_lower",{n_lay_out, NLAY1});
        this->layer1_bias_upper = grp.get_variable<float>("bias1_upper",{NLAY1});
        this->output_bias_upper = grp.get_variable<float>("bias2_upper",{n_lay_out});
        this->layer1_wgth_upper = grp.get_variable<float>("wgth1_upper",{NLAY1,     n_lay_in});
        this->output_wgth_upper = grp.get_variable<float>("wgth2_upper",{n_lay_out, NLAY1});
    }
    else if constexpr (NLAYER == 2)
    {
        this->layer1_bias_lower = grp.get_variable<float>("bias1_lower",{NLAY1});
        this->layer2_bias_lower = grp.get_variable<float>("bias2_lower",{NLAY2});
        this->output_bias_lower = grp.get_variable<float>("bias3_lower",{n_lay_out});
        this->layer1_wgth_lower = grp.get_variable<float>("wgth1_lower",{NLAY1,     n_lay_in});
        this->layer2_wgth_lower = grp.get_variable<float>("wgth2_lower",{NLAY2,     NLAY1});
        this->output_wgth_lower = grp.get_variable<float>("wgth3_lower",{n_lay_out, NLAY2});
        this->layer1_bias_upper = grp.get_variable<float>("bias1_upper",{NLAY1});
        this->layer2_bias_upper = grp.get_variable<float>("bias2_upper",{NLAY2});
        this->output_bias_upper = grp.get_variable<float>("bias3_upper",{n_lay_out});
        this->layer1_wgth_upper = grp.get_variable<float>("wgth1_upper",{NLAY1,     n_lay_in});
        this->layer2_wgth_upper = grp.get_variable<float>("wgth2_upper",{NLAY2,     NLAY1});
        this->output_wgth_upper = grp.get_variable<float>("wgth3_upper",{n_lay_out, NLAY2});
    }
    else if constexpr (NLAYER == 3)
    {
        this->layer1_bias_lower = grp.get_variable<float>("bias1_lower",{NLAY1});
        this->layer2_bias_lower = grp.get_variable<float>("bias2_lower",{NLAY2});
        this->layer3_bias_lower = grp.get_variable<float>("bias3_lower",{NLAY3});
        this->output_bias_lower = grp.get_variable<float>("bias4_lower",{n_lay_out});
        this->layer1_wgth_lower = grp.get_variable<float>("wgth1_lower",{NLAY1,     n_lay_in});
        this->layer2_wgth_lower = grp.get_variable<float>("wgth2_lower",{NLAY2,     NLAY1});
        this->layer3_wgth_lower = grp.get_variable<float>("wgth3_lower",{NLAY3,     NLAY2});
        this->output_wgth_lower = grp.get_variable<float>("wgth4_lower",{n_lay_out, NLAY3});
        this->layer1_bias_upper = grp.get_variable<float>("bias1_upper",{NLAY1});
        this->layer2_bias_upper = grp.get_variable<float>("bias2_upper",{NLAY2});
        this->layer3_bias_upper = grp.get_variable<float>("bias3_upper",{NLAY3});
        this->output_bias_upper = grp.get_variable<float>("bias4_upper",{n_lay_out});
        this->layer1_wgth_upper = grp.get_variable<float>("wgth1_upper",{NLAY1,     n_lay_in});
        this->layer2_wgth_upper = grp.get_variable<float>("wgth2_upper",{NLAY2,     NLAY1});
        this->layer3_wgth_upper = grp.get_variable<float>("wgth3_upper",{NLAY3,     NLAY2});
        this->output_wgth_upper = grp.get_variable<float>("wgth4_upper",{n_lay_out, NLAY3});
    }

    this->mean_input_lower   = grp.get_variable<float>("Fmean_lower",{n_lay_in});
    this->stdev_input_lower  = grp.get_variable<float>("Fstdv_lower",{n_lay_in});
    this->mean_output_lower  = grp.get_variable<float>("Lmean_lower",{n_lay_out});
    this->stdev_output_lower = grp.get_variable<float>("Lstdv_lower",{n_lay_out});
    
    this->mean_input_upper   = grp.get_variable<float>("Fmean_upper",{n_lay_in});
    this->stdev_input_upper  = grp.get_variable<float>("Fstdv_upper",{n_lay_in});
    this->mean_output_upper  = grp.get_variable<float>("Lmean_upper",{n_lay_out});
    this->stdev_output_upper = grp.get_variable<float>("Lstdv_upper",{n_lay_out});

}

template class Network<NLAYER, NLAY1, NLAY2, NLAY3>;



