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

vector<vector<float>> Dense::backward(vector<vector<float>> x)
{
    return x;
}
