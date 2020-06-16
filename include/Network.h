#ifndef NETWORK_H
#define NETWORK_H

#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>

class Network
{
    public:
        void inference(
            float* inputs,
            float* outputs,
            const int n_batch,
            const int lower_atmos,
            const int do_exp,
            const int do_norm,
            const int n_layers,
            const int n_layer1,
            const int n_layer2,
            const int n_layer3) const;

        Network ();
        Network(Netcdf_group& grp,
                const int n_layers,
                const int n_layer1,
                const int n_layer2,
                const int n_layer3,
                const int n_layer_out,
                const int n_layer_in);

    private:
        int n_batch_lower;
        int n_batch_upper;
        int n_layer_out;
        int n_layer_in;

        //all weights and biases of the different networks
        std::vector<float> output_wgth_lower;
        std::vector<float> output_wgth_upper;
        std::vector<float> output_bias_lower;
        std::vector<float> output_bias_upper;

        std::vector<float> layer1_wgth_lower;
        std::vector<float> layer1_bias_lower;
        std::vector<float> layer1_wgth_upper;
        std::vector<float> layer1_bias_upper;

        std::vector<float> layer2_wgth_lower;
        std::vector<float> layer2_bias_lower;
        std::vector<float> layer2_wgth_upper;
        std::vector<float> layer2_bias_upper;

        std::vector<float> layer3_wgth_lower;
        std::vector<float> layer3_bias_lower;
        std::vector<float> layer3_wgth_upper;
        std::vector<float> layer3_bias_upper;

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
