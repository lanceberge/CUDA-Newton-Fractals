#include "polynomial.h"
#include "complex.h"

// find the first derivative of a polynomial
Polynomial derivative(Polynomial P)
{
    int order = P.order;

    Polynomial Pprime;
    Pprime.coeffs = (dfloat *)calloc(order, sizeof(dfloat));
    Pprime.order = order - 1;

    dfloat *coeffs = P.coeffs;

    // update Pprime coeffs
    for (int i = 0; i < order; ++i)
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
        int coeff = coeffs[i];

        // zPow = z, then z^2, then z^3, etc.
        ReSum += coeff*zPow.Re;
        ImSum += coeff*zPow.Im;

        // update zPow to zPow*zPow
        zPow = cMul(zPow, z);
    }

    Complex P_z = {ReSum, ImSum};

    return P_z;
}

void printP(Polynomial P)
{
    dfloat *coeffs = P.coeffs;
    int order      = P.order;

    printf("Order: %d\n", order);

    for (int i = 0; i < order; ++i)
        printf("%f*z^%d + ", coeffs[i], order-i);

    printf("%f\n", coeffs[order]);

}
