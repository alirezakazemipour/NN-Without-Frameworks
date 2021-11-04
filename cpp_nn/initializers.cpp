#include "initializers.h"


float_batch Constant::initialize(unsigned int w, unsigned int h)
{
    float_batch temp(w, vector<float>(h, 1));

    for(unsigned int i  = 0; i < w; i++){
        for(unsigned int j = 0; j < h; j++){
            temp[i][j] = this->c;
        }
    }
    return temp;
}

float_batch RandomUniform::initialize(unsigned int w, unsigned int h){
    float_batch temp(w, vector<float>(h, 1));
    std::mt19937 gen(this->rd());
    for(unsigned int i  = 0; i < w; i++){
        for(unsigned int j = 0; j < h; j++){
            temp[i][j] = this->dis(gen);
        }
    }
    return temp;
}
