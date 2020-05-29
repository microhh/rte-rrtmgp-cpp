#ifndef NETWORK_H
#define NETWORK_H

#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>

constexpr int layer_count  = 0;  //number of layers used
constexpr int input_gases  = 0;  //number of extra gases
constexpr int layer1_nodes = 64; //nodes in first layer (if used)
constexpr int layer2_nodes = 64; //nodes in second layer (if used)
constexpr int layer3_nodes = 0;  //nodes in third layer (if used)

#ifdef EXTRA_GASES_IN
    constexpr int n_gas   = EXTRA_GASES_IN;
#else
    constexpr int n_gas   = 0; //default: on extra gases (only water vapour);
#endif

#ifdef NLAY1
    constexpr int n_lay1  = NLAY1;
#else
    constexpr int n_lay1  = 0; //default: 0 neurons in first layer
#endif

#ifdef NLAY2
    constexpr int n_lay2  = NLAY2;
#else
    constexpr int n_lay2  = 0; //default: 0 neurons in second layer
#endif

#ifdef NLAY3
    constexpr int n_lay3  = NLAY3;
#else
    constexpr int Nlay3   = 0; //default: 0 neurons in third layer
#endif

#ifdef NLAYER
    constexpr int n_layer = NLAYER;
#else
    constexpr int n_layer = 0; //default: 0 layers (linear network, i.e. linear regression)
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
