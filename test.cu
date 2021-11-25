// Test Polynomial
#include "newton.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    // Polynomial Tests
    Polynomial P;
    Polynomial Pprime;

    // create a polynomial
    int order = 4;

    P.order = order;

    dfloat coeffs[5] = {1, 2, 3, 4, 5};
    P.coeffs = coeffs;

    Pprime = derivative(P);

    printf("Pprime:\n");

    for (int i = 0; i < Pprime.order; ++i)
    {
        if (i != Pprime.order - 1)
        // coeffs[i]*z^order-i
            printf("%f*z^%d + ", coeffs[i], order-i);

        else
            printf("%f*z^d\n", coeffs[i], order-i);
    }

    return 0;
}
// Test complex
