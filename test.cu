// Test Polynomial
#include "newton.h"

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

    printP(P);
    printf("Pprime:\n");
    printP(Pprime);

    return 0;
}
// Test complex
