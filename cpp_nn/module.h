#ifndef MODULE_H
#define MODULE_H

#include <vector>

using namespace std;


class Module
{
public:
    virtual  vector<vector<float>> forward(const vector<vector<float>> &input)=0;
};

#endif // MODULE_H
