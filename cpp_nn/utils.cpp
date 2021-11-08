#include "utils.h"
#include <iostream>

float_batch Utils::mat_mul(float_batch A, float_batch B)
{
    unsigned int n = A.size(), k = B.size();
    unsigned int m = A[0].size(), l = B[0].size();

    if(m != k){
        throw "Invalid matrices' shapes!";
    }
    float_batch temp(n, vector<float>(l, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < l; j++){
            temp[i][j] = 0;
            for(unsigned int r = 0; r < k; r++){
                temp[i][j] += A[i][r] * B[r][j];
            }
        }
    }
    return temp;
}

float_batch Utils::element_wise_mul(float_batch A, float_batch B)
{
    unsigned int n = A.size(), k = B.size();
    unsigned int m = A[0].size(), l = B[0].size();

    if(n != k || m != l){
        throw "Invalid matrices' shapes!";
    }
    float_batch temp(n, vector<float>(l, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < l; j++){
            temp[i][j] = A[i][j] * B[i][j];
        }
    }
    return temp;
}

float_batch Utils::mat_add(float_batch A, float_batch B)
{
    unsigned int n = A.size(), k = B.size();
    unsigned int m = A[0].size(), l = B[0].size();

    if(n != k || m != l){
        throw "Invalid matrices' shapes!";
    }
    float_batch temp(n, vector<float>(l, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < l; j++){
            temp[i][j] = A[i][j] + B[i][j];
        }
    }
    return temp;
}


float_batch Utils::rescale(float_batch A, float scale)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(n, vector<float>(m, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < m; j++){
            temp[i][j] = A[i][j] * scale;
        }
    }
    return temp;
}

float_batch Utils::add_scalar(float_batch A, float scalar)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(n, vector<float>(m, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < m; j++){
            temp[i][j] = A[i][j] + scalar;
        }
    }
    return temp;
}

float_batch Utils::transpose(float_batch A)
{
    unsigned int n = A.size(), m = A[0].size();

    float_batch temp(m, vector<float>(n, 1));
    for(unsigned int i = 0; i < m; i++){
        for(unsigned int j = 0; j < n; j++){
            temp[i][j] = A[j][i];
        }
    }
    return temp;

}

float_batch Utils::element_wise_sqrt(float_batch A)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(n, vector<float>(m, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < m; j++){
            temp[i][j] = sqrt(A[i][j]);
        }
    }
    return temp;

}

float_batch Utils::element_wise_rev(float_batch A)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(n, vector<float>(m, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < m; j++){
            temp[i][j] =  1 / A[i][j];
        }
    }
    return temp;

}
