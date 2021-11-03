#include "layers.h"


vector<vector<float>> Dense::forward(const vector<vector<float> > &x)
{
    this->input = x;
    vector<vector<float>> z = this->utils.mat_mul(x, this->W);
    int x_dim0 = x.size();
    vector<vector<float>> b(x_dim0, vector<float>(this->b[0].size()));
    for(int i = 0; i < x_dim0; i++){
        b[i] = this->b[0];
    }
    z = this->utils.mat_add(z, b);
    this->z = z;
    vector<vector<float>> a = (!this->activation.compare("relu")) ? this->relu.forward(z) : this->linear.forward(z);

    return a;
}

vector<vector<float>> Dense::backward(vector<vector<float>> &delta)
{
    vector<vector<float> > dz;
    if(!this->activation.compare("relu")){
        dz = this->utils.element_wise_mul(delta, this->relu.derivative(this->z));
    }
    else{
        dz = this->utils.element_wise_mul(delta, this->linear.derivative(this->z));
    }
    vector<vector<float> > input_t = this->utils.transpose(this->input);
    vector<vector<float> > dw = this->utils.mat_mul(input_t, dz);
    this->dW = this->utils.rescale(dw, 1.0 / dz.size());

    vector<vector<float> > ones_t(1, vector<float>(dz.size(), 1));
    for(size_t i = 0; i < ones_t.size(); i++){
        for(size_t j = 0; j < ones_t[0].size(); j++){
            ones_t[i][j] = 1;
        }
    }
    vector<vector<float> > db = this->utils.mat_mul(ones_t, dz);
    this->db = this->utils.rescale(db, 1.0 / dz.size());


    vector<vector<float> > w_t = this->utils.transpose(this->W);
    delta = this->utils.mat_mul(dz, w_t);
    return delta;
}
