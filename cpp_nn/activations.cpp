#include "activations.h"


vector<vector<float>> Linear::forward(const vector<vector<float> > &x)
{
    return x;

}

vector<vector<float>> Linear::derivative(const vector<vector<float> > &x)
{
    int w = x.size(), h = x[0].size();
    vector<vector<float>> temp(w, vector<float>(h, 1));

    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            temp[i][j] = 1;
        }
    }
    return temp;
}

vector<vector<float>> ReLU::forward(const vector<vector<float> > &x)
{
    int w = x.size(), h = x[0].size();
    vector<vector<float>> temp(w, vector<float>(h, 1));

    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            temp[i][j] = (x[i][j] > 0) ? x[i][j] : 0;
        }
    }
    return temp;
}

vector<vector<float>> ReLU::derivative(const vector<vector<float> > &x)
{
    int w = x.size(), h = x[0].size();
    vector<vector<float>> temp(w, vector<float>(h, 1));

    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            temp[i][j] = (x[i][j] > 0) ? 1 : 0;
        }
    }
    return temp;
}
