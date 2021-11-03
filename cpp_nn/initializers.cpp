#include "initializers.h"


vector<vector<float>> Constant::initialize(unsigned int w, unsigned int h)
{
    vector<vector<float>> temp(w, vector<float>(h, 1));

    for(unsigned int i  = 0; i < w; i++){
        for(unsigned int j = 0; j < h; j++){
            temp[i][j] = this->c;
        }
    }
    return temp;
}

vector<vector<float>> RandomUniform::initialize(unsigned int w, unsigned int h){
    vector<vector<float>> temp(w, vector<float>(h, 1));    std::mt19937 gen(this->rd());
    for(unsigned int i  = 0; i < w; i++){
        for(unsigned int j = 0; j < h; j++){
            temp[i][j] = this->dis(gen);
        }
    }
    return temp;
}
