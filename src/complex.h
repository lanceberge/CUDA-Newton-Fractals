#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#define dfloat double

#include "cuda.h"
#include <stdio.h>

struct Complex
{
    dfloat Re;
    dfloat Im;

    // return the product of two complex numbers
    __host__ __device__ Complex operator*(Complex z2);

    // subtract two complex numbers
    __device__ Complex operator-(Complex z2);

    // divide two complex numbers
    __device__ Complex operator/(Complex z2);

    // print to stdout
    void printComplex(Complex z);
};

#endif
