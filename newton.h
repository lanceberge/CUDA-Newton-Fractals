#ifndef __NEWTON_H__
#define __NEWTON_H__

#include "polynomial.h"

// fill arrays for points before and after performing the newton iteration on them
__global__ void fillArrays(int ReSpacing, int ImSpacing, Complex *zValsInitial,
                Complex *zVals, int NRe, int NIm);

// perform Nit iterations of newton's method on a polynomial p
__global__ void newtonIterate(Complex *zVals, Polynomial P, Polynomial Pprime,
                              int N, int Nit);

__host__ __device__ int findSolns(Complex *solns, Complex *zVals,
                                   int nSolns, int nVals);
// L2 distance between two points
__host__ __device__ dfloat L2Distance(Complex z1, Complex z2);

void outputToCSV(const char *filename, int N, Complex *zVals, int *closest);

void outputSolnsToCSV(const char *filename, int nSolns, Complex *solns);

#endif
