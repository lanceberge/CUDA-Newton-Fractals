#include "complex.h"

// return the product of two complex numbers
__device__ Complex Complex::operator*(Complex z2)
{
    dfloat Re = this->Re*z2.Re - this->Im*z2.Im;
    dfloat Im = this->Re*z2.Im + this->Im*z2.Re;

    Complex z = {Re, Im};
    return z;
}

// subtract two complex numbers
__device__ Complex Complex::operator-(Complex z2)
{
    Complex z = {this->Re - z2.Re, this->Im - z2.Im};
    return z;
}

// divide two complex numbers - implementation from
// https://pixel.ecn.purdue.edu:8443/purpl/WSJ/projects/DirectionalStippling/include/cuComplex.h
__device__ Complex Complex::operator/(Complex z2)
{
    dfloat s = (fabs(z2.Re)) + (fabs(z2.Im));
    dfloat oos = 1.0 / s;
    dfloat ars = this->Re * oos;
    dfloat ais = this->Im * oos;
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
void Complex::printComplex(Complex z)
{
    printf("Re: %f, Im: %f\n", z.Re, z.Im);
}
