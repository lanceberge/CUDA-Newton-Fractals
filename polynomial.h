#ifndef __POLYNOMIAL_H__
#define __POLYNOMIAL_H__

#include "point.h"

// a struct representing polynomials - i.e. the polynomial Ax^3 + Cx + D
// would have order=3 and coeffs=[A,0,C,D]
typedef struct Polynomial
{
    int order;
    dfloat *coeffs;

} Polynomial;

// return the first derivative of a polynomial
Polynomial *derivative(Polynomial *P);

// find P(z) - plug in a point z to the polynomial
Point *Pz(Polynomial *P, Point *z);

#endif
