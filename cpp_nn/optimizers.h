#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <layers.h>
#include <utils.h>
#include <math.h>

# define eps 0.00000001

using namespace std;

class Optimizer
{
public:
    float lr;
    vector<Dense*> parameters;

    Optimizer(float lr, vector<Dense*> &params);
    virtual void apply()=0;
};

class SGD : public Optimizer
{
public:
    Utils utils;
    SGD(float lr, vector<Dense*> &params) : Optimizer{lr, params}{};
    void apply();

};

class Momentum : public Optimizer
{
public:
    Utils utils;
    float mu;
    vector<float_batch> gW, gb;
    Momentum(float lr, float mu, vector<Dense*> &params);
    void apply();

};

class RMSProp : public Optimizer
{
public:
    Utils utils;
    float beta=0.99;

    vector<float_batch> sW, sb;
    RMSProp(float lr, float beta, vector<Dense*> &params);
    void apply();

};

class AdaGrad : public Optimizer
{
public:
    Utils utils;
    vector<float_batch> sW, sb;
    AdaGrad(float lr, vector<Dense*> &params);
    void apply();

};
#endif // OPTIMIZER_H
