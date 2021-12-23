#include "complex.h"

// return the product of two complex numbers
__device__ Complex cMul(Complex z1, Complex z2)
{
    dfloat Re = z1.Re*z2.Re - z1.Im*z2.Im;
    dfloat Im = z1.Re*z2.Im + z1.Im*z2.Re;

    Complex z = {Re, Im};
    return z;
}

// subtract two complex numbers
__device__ Complex cSub(Complex z1, Complex z2)
{
    Complex z = {z1.Re - z2.Re, z1.Im - z2.Im};
    return z;
}

// divide two complex numbers - implementation from
// https://pixel.ecn.purdue.edu:8443/purpl/WSJ/projects/DirectionalStippling/include/cuComplex.h
__device__ Complex cDiv(Complex z1, Complex z2)
{
    dfloat s = (fabs(z2.Re)) + (fabs(z2.Im));
    dfloat oos = 1.0 / s;
    dfloat ars = z1.Re * oos;
    dfloat ais = z1.Im * oos;
    dfloat brs = z2.Re * oos;
    dfloat bis = z2.Im * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;

    dfloat Re = ((ars * brs) + (ais * bis)) * oos;
    dfloat Im = ((ais * brs) - (ars * bis)) * oos;

    Complex z = {Re, Im};

    return z;
}

// print a complex number
void printComplex(Complex z)
{
    printf("Re: %f, Im: %f\n", z.Re, z.Im);
}
