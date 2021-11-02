#include "constant.h"

constant::constant(float c)
{
    this->c = c;

}

float **constant::initialize(unsigned int w, unsigned int h)
{
    float **temp = new float*[w];
    for(unsigned int i = 0; i < w; i++){
        temp[w] = new float[h];
        for(unsigned int j = 0; j < h; j++){
            temp[i][j] = this->c;
        }
    }
    return temp;
}
