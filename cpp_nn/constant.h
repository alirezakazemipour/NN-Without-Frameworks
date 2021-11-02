#ifndef CONSTANT_H
#define CONSTANT_H

#include <initializers.h>


class constant : public Initilizer
{
private:
    float c = 0;
public:
    constant(float c);
    float** initialize(unsigned int w, unsigned int h);

};

#endif // CONSTANT_H
