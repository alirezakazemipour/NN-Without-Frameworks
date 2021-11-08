#ifndef LAYERS_H
#define LAYERS_H

#include <initializers.h>
#include <utils.h>
#include <activations.h>
#include <array>

using namespace std;

class Layer{

public:
    virtual float_batch forward(const float_batch &x) = 0;
    virtual float_batch backward(float_batch &x) = 0;
};


class Dense : public Layer{

public:
    float_batch W, dW, b, db;
    int in_features, out_features;
    std::string weight_initializer;
    std::string bias_initializer;
    std::string activation = "linear";
    float_batch input, z;

    RandomUniform random_uniform;
    Constant zeros{0.0};
    XavierUniform xavier_uniform;
    Utils utils;
    Linear linear;
    ReLU relu;

    Dense(int in_features, int out_features, std::string activation, std::string weight_initializer, std::string bias_initialzer){
        this->in_features = in_features;
        this->out_features = out_features;
        this->weight_initializer = weight_initializer;
        this->bias_initializer = bias_initialzer;
        this->activation = activation;

        HeNormal he_normal(this->activation, "fan_in");

        if (!weight_initializer.compare("xavier_uniform")){
            this->W = this->xavier_uniform.initialize(this->in_features, this->out_features);
        }
        else if (!weight_initializer.compare("he_normal")){
            this->W = he_normal.initialize(this->in_features, this->out_features);
        }
        else{
            this->W = this->random_uniform.initialize(this->in_features, this->out_features);
        }
        if (!bias_initializer.compare("zeros")) {
            this->b = this->zeros.initialize(1, this->out_features);
        }

        this->dW = this->zeros.initialize(this->in_features, this->out_features);
        this->db = this->zeros.initialize(1, this->out_features);
    };



    // Layer interface
    float_batch forward(const float_batch &x);
    float_batch backward(float_batch &delta);
};

#endif // LAYERS_H
