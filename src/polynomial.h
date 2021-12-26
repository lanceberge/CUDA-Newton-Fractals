#pragma once

/* #include "cuComplex.h" */
#include "complex.h"

// a struct representing polynomials - i.e. the polynomial Ax^3 + Cx + D
// would have order=3 and coeffs=[A,0,C,D]
typedef struct Polynomial
{
    int order;
    dfloat *coeffs;

} Polynomial;

// return the first derivative of a polynomial
Polynomial derivative(const Polynomial& P);

// find P(z) - plug in a point z to the polynomial
__host__ __device__ Complex Pz(const Polynomial& P, const Complex& z);

// return h_P with device coeffs array
Polynomial deviceP(const Polynomial& h_P);

// return a random polynomial with a specified order, with coefficients
// random between -max and max. seed is the seed for drand
Polynomial randomPolynomial(int order, int max, int seed);

// print to stdout
void printP(const Polynomial& P);
