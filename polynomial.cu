#include "polynomial.h"
#include <complex.h>

// find the first derivative of a polynomial
Polynomial *derivative(Polynomial *P)
{
    Polynomial *Pprime = (Polynomial *)malloc(sizeof(Polynomial *));
    Prime->coeffs = (dfloat)malloc(order-1 * sizeof(dfloat));

    int order = P->order;
    dfloat *coeffs = P->coeffs;

    if (order < 1)
        return NULL;

    Pprime->order = order - 1;

    // update Pprime coeffs
    for (int i = 0; i < order - 1; ++i)
    {
        Pprime->coeffs[i] = coeffs[i]*(order-i);
    }

    return Pprime;
}

// find P(z) - plug in a point z to the polynomial
__host__ __device__ std::complex<dfloat> Pz(Polynomial *P, std::complex<dfloat> z)
{
    dfloat complex cumulativeSum = 0;

    // for A, B, C, D in coeffs. of P, return the cumulative sum of Az^4 + Bz^3 + ...
    for (int i = 0; i < P->order; ++i)
        cumulativeSum += std::pow(z, order-i);

    return cumulativeSum;
}

// free associated memory
void freePolynomial(Polynomial *P)
{
    free(P->coeffs);
    free(P);
}
