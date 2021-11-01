#include "polynomial.h"

// find the first derivative of a polynomial
Polynomial *derivative(Polynomial *P);
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

#endif
