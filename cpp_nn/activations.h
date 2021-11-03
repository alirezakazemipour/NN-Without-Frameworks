#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <iostream>
#include <vector>
#include <array>

using namespace std;

class Activation
{
public:
    virtual vector<vector<float>> forward(const vector<vector<float>> &x)=0;
    virtual vector<vector<float>> derivative(const vector<vector<float>> &x)=0;

};

class ReLU : public Activation
{
public:
    vector<vector<float>> forward(const vector<vector<float>> &x);
    vector<vector<float>> derivative(const vector<vector<float>> &x);
};

class Linear : public Activation
{
public:
    vector<vector<float>> forward(const vector<vector<float>> &x);
    vector<vector<float>> derivative(const vector<vector<float>> &x);
};


#endif // ACTIVATIONS_H
