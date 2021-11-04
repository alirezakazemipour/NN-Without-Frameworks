#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include <random>
#include <vector>

using namespace std;

typedef vector<vector<float>> float_batch;


class Initializer{
public:
    virtual float_batch initialize(unsigned int w, unsigned int h)=0;
};

class Constant : public Initializer{
public:
    Constant(float c){
        this->c = c;
    }
    float_batch initialize(unsigned int w, unsigned int h);

private:
    float c = 0;
};

class RandomUniform : public Initializer{
public:
    float_batch initialize(unsigned int w, unsigned int h);

private:
    std::random_device rd;
    std::uniform_real_distribution<float> dis{0.0, 1.0};


};

#endif // INITIALIZERS_H
