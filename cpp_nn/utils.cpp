#include "utils.h"
#include <iostream>

vector<vector<float>> Utils::mat_mul(vector<vector<float>> A, vector<vector<float>> B)
{
    unsigned int n = A.size(), k = B.size();
    unsigned int m = A[0].size(), l = B[0].size();

    if(m != k){
        throw "Invalid matrices' shapes!";
    }
    vector<vector<float>> temp(n, vector<float>(l, 1));
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

vector<vector<float>> Utils::element_wise_mul(vector<vector<float>> A, vector<vector<float> > B)
{
    unsigned int n = A.size(), k = B.size();
    unsigned int m = A[0].size(), l = B[0].size();

    if(n != k || m != l){
        throw "Invalid matrices' shapes!";
    }
    vector<vector<float>> temp(n, vector<float>(l, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < l; j++){
            temp[i][j] = A[i][j] * B[i][j];
        }
    }
    return temp;
}

vector<vector<float>> Utils::mat_add(vector<vector<float>> A, vector<vector<float>> B)
{
    unsigned int n = A.size(), k = B.size();
    unsigned int m = A[0].size(), l = B[0].size();

    if(n != k || m != l){
        throw "Invalid matrices' shapes!";
    }
    vector<vector<float>> temp(n, vector<float>(l, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < l; j++){
            temp[i][j] = A[i][j] + B[i][j];
        }
    }
    return temp;
}


vector<vector<float> > Utils::rescale(vector<vector<float> > A, float scale)
{
    unsigned int n = A.size(), m = A[0].size();
    vector<vector<float>> temp(n, vector<float>(m, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < m; j++){
            temp[i][j] = A[i][j] * scale;
        }
    }
    return temp;
}

vector<vector<float> > Utils::transpose(vector<vector<float> > A)
{
    unsigned int n = A.size(), m = A[0].size();

    vector<vector<float>> temp(m, vector<float>(n, 1));
    for(unsigned int i = 0; i < m; i++){
        for(unsigned int j = 0; j < n; j++){
            temp[i][j] = A[j][i];
        }
    }
    return temp;

}
