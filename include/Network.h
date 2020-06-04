#ifndef NETWORK_H
#define NETWORK_H

#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>

#ifdef NGASES
    constexpr int NGAS   = NGASES;
#else
    constexpr int NGAS   = 0; //default: on extra gases (only water vapour);
#endif

#ifdef NLAYER1
    constexpr int NLAY1  = NLAYER1;
#else
    constexpr int NLAY1  = 0; //default: 0 neurons in first layer
#endif

#ifdef NLAYER2
    constexpr int NLAY2  = NLAYER2;
#else
    constexpr int NLAY2  = 0; //default: 0 neurons in second layer
#endif

#ifdef NLAYER3
    constexpr int NLAY3  = NLAYER3;
#else
    constexpr int NLAY3  = 0; //default: 0 neurons in third layer
#endif

#ifdef NLAYERS
    constexpr int NLAYER = NLAYERS;
#else
    constexpr int NLAYER = 0; //default: 0 layers (linear network, i.e. linear regression)
#endif

template <int NLAYER, int NLAY1, int NLAY2, int NLAY3>
class Network
{
    public:
        void inference(
            float* inputs,
            float* outputs,
            const int lower_atmos,
            const int do_exp,
            const int do_inpnorm);

        Network(const int n_batch_lower,
                const int n_batch_upper,
                Netcdf_group& grp,
                const int n_lay_out,
                const int n_lay_in);

    private:
        int n_batch_lower;
        int n_batch_upper;
        int n_lay_out;
        int n_lay_in;

        //all weights and biases of the different networks
        std::vector<float> output_wgth_lower;
        std::vector<float> output_wgth_upper;
        std::vector<float> output_bias_lower;
        std::vector<float> output_bias_upper;

        std::vector<float> layer1_wgth_lower;
        std::vector<float> layer1_bias_lower;
        std::vector<float> layer1_wgth_upper;
        std::vector<float> layer1_bias_upper;
        std::vector<float> layer1;

        std::vector<float> layer2_wgth_lower;
        std::vector<float> layer2_bias_lower;
        std::vector<float> layer2_wgth_upper;
        std::vector<float> layer2_bias_upper;
        std::vector<float> layer2;
        
        std::vector<float> layer3_wgth_lower;
        std::vector<float> layer3_bias_lower;
        std::vector<float> layer3_wgth_upper;
        std::vector<float> layer3_bias_upper;
        std::vector<float> layer3;

        //means and standard deviations to (de)normalize inputs and optical properties
        std::vector<float> mean_input_lower;
        std::vector<float> stdev_input_lower;
        std::vector<float> mean_output_lower;
        std::vector<float> stdev_output_lower;

        std::vector<float> mean_input_upper;
        std::vector<float> stdev_input_upper;
        std::vector<float> mean_output_upper;
        std::vector<float> stdev_output_upper;
};
#endif
