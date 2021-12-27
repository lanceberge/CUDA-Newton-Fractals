#pragma once

/* #include "cuComplex.h" */
#include "complex.h"

// a struct representing polynomials - i.e. the polynomial Ax^3 + Cx + D
// would have order=3 and coeffs=[A,0,C,D]
struct Polynomial
{
    int order;
    dfloat *h_coeffs;
    dfloat *c_coeffs;

    // coeffs is an array of size order
    Polynomial(int order, dfloat *h_coeffs);

    // copy constructor
    Polynomial(const Polynomial& p);

    // return the first derivative of a polynomial
    Polynomial derivative();

    // find P(z) - plug in a point z to the polynomial
    __device__ Complex c_Pz(const Complex& z) const;

    // host version of P(z)
    __host__ Complex h_Pz(const Complex& z) const;

    ~Polynomial();
};

// return the coeffs to a random polynomial, each coeff is between
// -max and max. seed is the seed for drand
dfloat *randomCoeffs(int order, int max, int seed);
