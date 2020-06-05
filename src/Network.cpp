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
            float* restrict const hiddenlayer1,
            float* restrict const hiddenlayer2,
            float* restrict const hiddenlayer3,
            const int n_batch,
            const int n_lay_out,
            const int n_lay_in,
            const int do_exp,
            const int do_norm,
            const int n_layers,
            const int n_layer1,
            const int n_layer2,
            const int n_layer3)
    {  
        if (do_norm) {normalize_input(input, input_mean, input_stdev, n_batch, n_lay_in);}

        if (n_layers==0) //Linear regression
        {   
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_lay_out, n_batch, n_lay_in, 1.f,
                        output_wgth, n_lay_in, input, n_batch, 0.f, output, n_batch);
        }
        if (n_layers==1)
        {   
            matmul_bias_act_blas(n_batch, n_layer1, n_lay_in, layer1_wgth, layer1_bias, input, hiddenlayer1);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_lay_out, n_batch, n_layer1, 1.f,
                        output_wgth, n_layer1, hiddenlayer1, n_batch, 0.f, output , n_batch);
        }
        if (n_layers==2)
        {
            matmul_bias_act_blas(n_batch, n_layer1, n_lay_in, layer1_wgth, layer1_bias, input,  hiddenlayer1);
            matmul_bias_act_blas(n_batch, n_layer2, n_layer1,    layer2_wgth, layer2_bias, hiddenlayer1, hiddenlayer2);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_lay_out, n_batch, n_layer2, 1.f,
                        output_wgth, n_layer2, hiddenlayer2, n_batch, 0.f, output, n_batch);
        }
        if (n_layers==3)
        {
            matmul_bias_act_blas(n_batch, n_layer1, n_lay_in, layer1_wgth, layer1_bias, input,  hiddenlayer1);
            matmul_bias_act_blas(n_batch, n_layer2, n_layer1,    layer2_wgth, layer2_bias, hiddenlayer1, hiddenlayer2);
            matmul_bias_act_blas(n_batch, n_layer3, n_layer2,    layer3_wgth, layer3_bias, hiddenlayer2, hiddenlayer3);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_lay_out, n_batch, n_layer3, 1.f,
                        output_wgth, n_layer3, hiddenlayer3, n_batch, 0.f, output, n_batch);
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

void Network::inference(
        float* inputs,
        float* outputs,
        const int lower_atmos,
        const int do_exp,
        const int do_norm,
        const int n_layers,
        const int n_layer1,
        const int n_layer2,
        const int n_layer3) const
{
    std::vector<float> hiddenlayer1(std::max(n_layer1*this->n_batch_lower, n_layer1*this->n_batch_upper));
    std::vector<float> hiddenlayer2(std::max(n_layer2*this->n_batch_lower, n_layer2*this->n_batch_upper));
    std::vector<float> hiddenlayer3(std::max(n_layer3*this->n_batch_lower, n_layer3*this->n_batch_upper));
//    std::vector<float> Lhiddenlayer1(n_layer1*this->n_batch_lower);
//    std::vector<float> Lhiddenlayer2(n_layer2*this->n_batch_lower);
//    std::vector<float> Lhiddenlayer3(n_layer3*this->n_batch_lower);
//    std::vector<float> Uhiddenlayer1(n_layer1*this->n_batch_upper);
//    std::vector<float> Uhiddenlayer2(n_layer2*this->n_batch_upper);
//    std::vector<float> Uhiddenlayer3(n_layer3*this->n_batch_upper);
    if (lower_atmos == 1)
    {

        feedforward(
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
            hiddenlayer1.data(),//this->hiddenlayer1.data(),
            hiddenlayer2.data(),//this->hiddenlayer2.data(),
            hiddenlayer3.data(),//this->hiddenlayer3.data(),
            this->n_batch_lower,
            this->n_layer_out,
            this->n_layer_in,
            do_exp,
            do_norm,
            n_layers,
            n_layer1,
            n_layer2,
            n_layer3);
    }
    else
    {
//        if (n_layers>0) this->hiddenlayer1.resize(n_layer1 * this->n_batch_upper);
//        if (n_layers>1) this->hiddenlayer2.resize(n_layer2 * this->n_batch_upper);
//        if (n_layers>2) this->hiddenlayer3.resize(n_layer3 * this->n_batch_upper);
//        if (n_layers>0) HL1.resize(n_layer1 * this->n_batch_upper);
//        if (n_layers>1) HL2.resize(n_layer2 * this->n_batch_upper);
//        if (n_layers>2) HL3.resize(n_layer3 * this->n_batch_upper);
        feedforward(
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
            hiddenlayer1.data(),//this->hiddenlayer1.data(),
            hiddenlayer2.data(),//this->hiddenlayer2.data(),
            hiddenlayer3.data(),//this->hiddenlayer3.data(),
            this->n_batch_upper,
            this->n_layer_out,
            this->n_layer_in,
            do_exp,
            do_norm,
            n_layers,
            n_layer1,
            n_layer2,
            n_layer3);
    }

}

Network::Network(){}

Network::Network(const int n_batch_lower,
                 const int n_batch_upper,
                 Netcdf_group& grp,
                 const int n_layers,
                 const int n_layer1,
                 const int n_layer2,
                 const int n_layer3,
                 const int n_layer_out,
                 const int n_layer_in)
{
    this->n_batch_lower = n_batch_lower;
    this->n_batch_upper = n_batch_upper;
    this->n_layer_out   = n_layer_out;
    this->n_layer_in    = n_layer_in;

    if (n_layers == 0)
    {
        this->output_bias_lower = grp.get_variable<float>("bias1_lower",{n_layer_out});
        this->output_wgth_lower = grp.get_variable<float>("wgth1_lower",{n_layer_out, n_layer_in});
        this->output_bias_upper = grp.get_variable<float>("bias1_upper",{n_layer_out});
        this->output_wgth_upper = grp.get_variable<float>("wgth1_upper",{n_layer_out, n_layer_in});
    }
    else if (n_layers == 1)
    {
        this->layer1_bias_lower = grp.get_variable<float>("bias1_lower",{n_layer1});
        this->output_bias_lower = grp.get_variable<float>("bias2_lower",{n_layer_out});
        this->layer1_wgth_lower = grp.get_variable<float>("wgth1_lower",{n_layer1,    n_layer_in});
        this->output_wgth_lower = grp.get_variable<float>("wgth2_lower",{n_layer_out, n_layer1});
        this->layer1_bias_upper = grp.get_variable<float>("bias1_upper",{n_layer1});
        this->output_bias_upper = grp.get_variable<float>("bias2_upper",{n_layer_out});
        this->layer1_wgth_upper = grp.get_variable<float>("wgth1_upper",{n_layer1,    n_layer_in});
        this->output_wgth_upper = grp.get_variable<float>("wgth2_upper",{n_layer_out, n_layer1});
    }
    else if (n_layers == 2)
    {
        this->layer1_bias_lower = grp.get_variable<float>("bias1_lower",{n_layer1});
        this->layer2_bias_lower = grp.get_variable<float>("bias2_lower",{n_layer2});
        this->output_bias_lower = grp.get_variable<float>("bias3_lower",{n_layer_out});
        this->layer1_wgth_lower = grp.get_variable<float>("wgth1_lower",{n_layer1,    n_layer_in});
        this->layer2_wgth_lower = grp.get_variable<float>("wgth2_lower",{n_layer2,    n_layer1});
        this->output_wgth_lower = grp.get_variable<float>("wgth3_lower",{n_layer_out, n_layer2});
        this->layer1_bias_upper = grp.get_variable<float>("bias1_upper",{n_layer1});
        this->layer2_bias_upper = grp.get_variable<float>("bias2_upper",{n_layer2});
        this->output_bias_upper = grp.get_variable<float>("bias3_upper",{n_layer_out});
        this->layer1_wgth_upper = grp.get_variable<float>("wgth1_upper",{n_layer1,    n_layer_in});
        this->layer2_wgth_upper = grp.get_variable<float>("wgth2_upper",{n_layer2,    n_layer1});
        this->output_wgth_upper = grp.get_variable<float>("wgth3_upper",{n_layer_out, n_layer2});
    }
    else if (n_layers == 3)
    {
        this->layer1_bias_lower = grp.get_variable<float>("bias1_lower",{n_layer1});
        this->layer2_bias_lower = grp.get_variable<float>("bias2_lower",{n_layer2});
        this->layer3_bias_lower = grp.get_variable<float>("bias3_lower",{n_layer3});
        this->output_bias_lower = grp.get_variable<float>("bias4_lower",{n_layer_out});
        this->layer1_wgth_lower = grp.get_variable<float>("wgth1_lower",{n_layer1,    n_layer_in});
        this->layer2_wgth_lower = grp.get_variable<float>("wgth2_lower",{n_layer2,    n_layer1});
        this->layer3_wgth_lower = grp.get_variable<float>("wgth3_lower",{n_layer3,    n_layer2});
        this->output_wgth_lower = grp.get_variable<float>("wgth4_lower",{n_layer_out, n_layer3});
        this->layer1_bias_upper = grp.get_variable<float>("bias1_upper",{n_layer1});
        this->layer2_bias_upper = grp.get_variable<float>("bias2_upper",{n_layer2});
        this->layer3_bias_upper = grp.get_variable<float>("bias3_upper",{n_layer3});
        this->output_bias_upper = grp.get_variable<float>("bias4_upper",{n_layer_out});
        this->layer1_wgth_upper = grp.get_variable<float>("wgth1_upper",{n_layer1,    n_layer_in});
        this->layer2_wgth_upper = grp.get_variable<float>("wgth2_upper",{n_layer2,    n_layer1});
        this->layer3_wgth_upper = grp.get_variable<float>("wgth3_upper",{n_layer3,    n_layer2});
        this->output_wgth_upper = grp.get_variable<float>("wgth4_upper",{n_layer_out, n_layer3});
    }

    this->mean_input_lower   = grp.get_variable<float>("Fmean_lower",{n_layer_in});
    this->stdev_input_lower  = grp.get_variable<float>("Fstdv_lower",{n_layer_in});
    this->mean_output_lower  = grp.get_variable<float>("Lmean_lower",{n_layer_out});
    this->stdev_output_lower = grp.get_variable<float>("Lstdv_lower",{n_layer_out});
    
    this->mean_input_upper   = grp.get_variable<float>("Fmean_upper",{n_layer_in});
    this->stdev_input_upper  = grp.get_variable<float>("Fstdv_upper",{n_layer_in});
    this->mean_output_upper  = grp.get_variable<float>("Lmean_upper",{n_layer_out});
    this->stdev_output_upper = grp.get_variable<float>("Lstdv_upper",{n_layer_out});
}

