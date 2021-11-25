#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#define dfloat double

#include "cuda.h"

typedef struct Complex
{
    dfloat Re;
    dfloat Im;
} Complex;

// return the product of two complex numbers
__host__ __device__ Complex cMul(Complex z1, Complex z2);

// add two complex numbers
__host__ __device__ Complex cAdd(Complex z1, Complex z2);

// divide two complex numbers
__host__ __device__ Complex cDiv(Complex z1, Complex z2);

#endif
