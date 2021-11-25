#include "polynomial.h"
#include "complex.h"

// find the first derivative of a polynomial
Polynomial derivative(Polynomial P)
{
    int order = P.order;
    --order;

    Polynomial Pprime;
    Pprime.coeffs = (dfloat *)malloc(order * sizeof(dfloat));
    Pprime.order = order;

    dfloat *coeffs = P.coeffs;

    // update Pprime coeffs
    for (int i = 0; i < order - 1; ++i)
    {
        Pprime.coeffs[i] = coeffs[i]*(order-i);
    }

    return Pprime;
}

// find P(z) - plug in a point z to the polynomial
__host__ __device__ Complex Pz(Polynomial P, Complex z)
{
    dfloat *coeffs = P.coeffs;
    int order = P.order;

    dfloat ReSum = coeffs[order];
    dfloat ImSum = 0;

    // zPow on first iteration, then zPow^2, then ^3, etc.
    Complex zPow = {z.Re, z.Im};

    // for A, B, C, D in coeffs. of P, return the cumulative sum of Az^4 + Bz^3 + ...
    for (int i = order-1; i >= 0; --i)
    {
        int coeff = coeffs[order];

        // zPow = z, then z^2, then z^3, etc.
        ReSum += coeff*zPow.Re;
        ImSum += coeff*zPow.Im;

        // update zPow to zPow*zPow
        zPow = cMul(zPow, zPow);
    }

    Complex Pz = {ReSum, ImSum};

    return Pz;
}
