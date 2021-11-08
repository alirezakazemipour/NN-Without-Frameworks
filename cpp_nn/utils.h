#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <array>
#include <functional>
#include <math.h>

using namespace std;

typedef vector<vector<float>> float_batch;

class Utils
{
public:
    //    Utils();
    static float_batch mat_mul(float_batch A, float_batch B);
    static float_batch element_wise_mul(float_batch A, float_batch B);
    static float_batch mat_add(float_batch A, float_batch B);
    static float_batch rescale(float_batch A, float scale);
    static float_batch add_scalar(float_batch A, float scalar);
    static float_batch transpose(float_batch A);
    static float_batch element_wise_sqrt(float_batch A);
    static float_batch element_wise_rev(float_batch A);

};

#endif // UTILS_H
