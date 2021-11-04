#include "layers.h"


float_batch Dense::forward(const float_batch &x)
{
    this->input = x;
    float_batch z = this->utils.mat_mul(x, this->W);
    int x_dim0 = x.size();
    float_batch b(x_dim0, vector<float>(this->b[0].size()));
    for(int i = 0; i < x_dim0; i++){
        b[i] = this->b[0];
    }
    z = this->utils.mat_add(z, b);
    this->z = z;
    float_batch a = (!this->activation.compare("relu")) ? this->relu.forward(z) : this->linear.forward(z);

    return a;
}

float_batch Dense::backward(float_batch &delta)
{
    float_batch dz;
    if(!this->activation.compare("relu")){
        dz = this->utils.element_wise_mul(delta, this->relu.derivative(this->z));
    }
    else{
        dz = this->utils.element_wise_mul(delta, this->linear.derivative(this->z));
    }
    float_batch input_t = this->utils.transpose(this->input);
    float_batch dw = this->utils.mat_mul(input_t, dz);
    this->dW = this->utils.rescale(dw, 1.0 / dz.size());

    float_batch ones_t(1, vector<float>(dz.size(), 1));
    for(size_t i = 0; i < ones_t.size(); i++){
        for(size_t j = 0; j < ones_t[0].size(); j++){
            ones_t[i][j] = 1;
        }
    }
    float_batch db = this->utils.mat_mul(ones_t, dz);
    this->db = this->utils.rescale(db, 1.0 / dz.size());


    float_batch w_t = this->utils.transpose(this->W);
    delta = this->utils.mat_mul(dz, w_t);
    return delta;
}
