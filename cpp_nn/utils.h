#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <array>

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
    static float_batch transpose(float_batch A);
};

#endif // UTILS_H
