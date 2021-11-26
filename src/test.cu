// Test Polynomial
#include "newton.h"

int main(int argc, char **argv)
{
    // Polynomial Tests
    Polynomial P;
    Polynomial Pprime;

    // create a polynomial
    int order = 3;

    P.order = order;
    dfloat coeffs[4] = {-4, 6, 2, 2};

    P.coeffs = coeffs;

    Pprime = derivative(P);

    printP(P);
    printf("Pprime:\n");
    printP(Pprime);

    Complex z = {3, 2};
    Complex pz = Pz(P, z);
    printf("Pz:\n");

    // expected: 74, -108
    printComplex(pz);

    return 0;
}
