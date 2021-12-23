#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#define dfloat double

#include "cuda.h"
#include <stdio.h>

typedef struct Complex
{
    dfloat Re;
    dfloat Im;
} Complex;

// return the product of two complex numbers
__host__ __device__ Complex cMul(Complex z1, Complex z2);

// subtract two complex numbers
__device__ Complex cSub(Complex z1, Complex z2);

// divide two complex numbers
__device__ Complex cDiv(Complex z1, Complex z2);

// print to stdout
void printComplex(Complex z);

#endif
