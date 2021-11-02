#include "initializers.h"


float **Constant::initialize(unsigned int w, unsigned int h)
{
    float **temp = new float*[w];
    for(unsigned int i  = 0; i < w; i++){
        temp[i] = new float[h];
        for(unsigned int j = 0; j < h; j++){
            temp[i][j] = this->c;
        }
    }
    return temp;
}

float **RandomUniform::initialize(unsigned int w, unsigned int h){
    float** temp = new float* [w];
    std::mt19937 gen(this->rd());
    for(unsigned int i  = 0; i < w; i++){
        temp[i] = new float[h];
        for(unsigned int j = 0; j < h; j++){
            temp[i][j] = this->dis(gen);
        }
    }
    return temp;
}
