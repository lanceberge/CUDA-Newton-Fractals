#include "complex.h"

// return the product of two complex numbers
__host__ __device__ Complex cMul(Complex z1, Complex z2)
{
    dfloat Re = z1.Re*z2.Re - z1.Im*z2.Im;
    dfloat Im = z1.Re*z2.Im + z1.Im*z2.Re;

    Complex z = {Re, Im};
    return z;
}

// add two complex numbers
__host__ __device__ Complex cAdd(Complex z1, Complex z2)
{
    Complex z = {z1.Re + z2.Re, z1.Im + z2.Im};
    return z;
}

// divide two complex numbers
__host__ __device__ Complex cDiv(Complex z1, Complex z2)
{
    dfloat Re = (z1.Re*z2.Re + z1.Im*z2.Im)/(z2.Re*z2.Re + z2.Im*z2.Im);
    dfloat Im = (z2.Re*z1.Im - z1.Re*z2.Im)/(z2.Re*z2.Re + z2.Im*z2.Im);

    Complex z = {Re, Im};
    return z;
}
