#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <array>

using namespace std;

class Utils
{
public:
//    Utils();
    vector<vector<float>> mat_mul(vector<vector<float>> A, vector<vector<float>> B);
    vector<vector<float>> element_wise_mul(vector<vector<float>> A, vector<vector<float>> B);
    vector<vector<float>> mat_add(vector<vector<float>> A, vector<vector<float>> B);
    vector<vector<float>> rescale(vector<vector<float>> A, float scale);
    vector<vector<float>> transpose(vector<vector<float>> A);
};

#endif // UTILS_H
