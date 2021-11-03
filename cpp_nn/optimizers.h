#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <layers.h>
#include <utils.h>

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

#endif // OPTIMIZER_H
