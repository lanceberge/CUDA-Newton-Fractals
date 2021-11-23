#ifndef __POLYNOMIAL_H__
#define __POLYNOMIAL_H__

#define dfloat double

#include <complex>
#include "cuda.h"

// a struct representing polynomials - i.e. the polynomial Ax^3 + Cx + D
// would have order=3 and coeffs=[A,0,C,D]
typedef struct Polynomial
{
    int order;
    int *coeffs;

} Polynomial;

// return the first derivative of a polynomial
Polynomial *derivative(Polynomial *P);

// find P(z) - plug in a point z to the polynomial
__host__ __device__ std::complex<dfloat> Pz(Polynomial *P, std::complex<dfloat> z);

void freePolynomial(Polynomial *P);

#endif
