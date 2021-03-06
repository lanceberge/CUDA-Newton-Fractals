#pragma once

typedef double dfloat;

#include "cuda.h"
#include <stdio.h>

struct Complex
{
    dfloat Re;
    dfloat Im;

    __host__ __device__ Complex(dfloat x, dfloat y);

    // return the product of two complex numbers
    __host__ __device__ Complex operator*(const Complex& z2);

    // subtract two complex numbers
    __device__ Complex operator-(const Complex& z2);

    // divide two complex numbers
    __device__ Complex operator/(const Complex& z2);

    __host__ __device__ bool operator==(const Complex& z2);

    __host__ __device__ bool operator!=(const Complex& z2);

    __host__ __device__ bool isRoot();

    // print to stdout
    __host__ __device__ void printComplex();
};
