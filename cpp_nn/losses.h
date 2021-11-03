#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <math.h>
#include <utils.h>

using namespace std;

class Loss
{
public:
    float value;
    vector<vector<float>> delta;
    Loss(float value, vector<vector<float> > delta);
};

class LossFunc
{
public:
    vector<vector<float> > target, pred;
    virtual Loss apply(vector<vector<float> > pred, vector<vector<float> > target)=0;
    virtual vector<vector<float> > delta()=0;
};

class MSELoss : LossFunc
{
public:
    Utils utils;
    Loss apply(vector<vector<float> > pred, vector<vector<float> > target);
    vector<vector<float> > delta();
};
#endif // LOSS_H
