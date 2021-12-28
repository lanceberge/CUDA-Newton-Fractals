#include "complex.h"

__host__ __device__ Complex::Complex(dfloat x, dfloat y) : Re(x), Im(y) {}

// return the product of two complex numbers
__host__ __device__ Complex Complex::operator*(const Complex& z2)
{
    dfloat x = Re*z2.Re - Im*z2.Im;
    dfloat y = Re*z2.Im + Im*z2.Re;

    return Complex(x, y);
}

// subtract two complex numbers
__device__ Complex Complex::operator-(const Complex& z2)
{
    return Complex(Re - z2.Re, Im - z2.Im);
}

// divide two complex numbers - implementation from
// https://pixel.ecn.purdue.edu:8443/purpl/WSJ/projects/DirectionalStippling/include/cuComplex.h
__device__ Complex Complex::operator/(const Complex& z2)
{
    dfloat s = (fabs(z2.Re)) + (fabs(z2.Im));
    dfloat oos = 1.0 / s;
    dfloat ars = Re * oos;
    dfloat ais = Im * oos;
    dfloat brs = z2.Re * oos;
    dfloat bis = z2.Im * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;

    dfloat Re = ((ars * brs) + (ais * bis)) * oos;
    dfloat Im = ((ais * brs) - (ars * bis)) * oos;

    return Complex(Re, Im);
}

// print a complex number
void Complex::printComplex(const Complex& z)
{
    printf("Re: %f, Im: %f\n", z.Re, z.Im);
}
