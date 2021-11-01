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
Point *Pz(Polynomial *P, Point *z);
{
    dfloat complex z = P->Re + P->Im*I;

    dfloat complex cumulativeSum = 0;

    // for A, B, C, D in coeffs. of polynomial P,
    // return the cumulative sum of Az^4 + Bz^3 + ...
    // in a new Point struct
    for (int i = 0; i < P->order; ++i)
    {
        cumulativeSum += cpow(z, order-i);
    }

    z = cumulativeSum;

    Point *Pz = malloc(sizeof(*Pz));

    Pz->Re = creal(cumulativeSum);
    Pz->Im = cimage(cumulativeSum);
}

#endif
