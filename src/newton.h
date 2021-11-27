#ifndef __NEWTON_H__
#define __NEWTON_H__

#include "polynomial.h"

// fill arrays for points before and after performing the newton iteration on them
__global__ void fillArrays(dfloat ReSpacing, dfloat ImSpacing, Complex *zValsInitial,
                Complex *zVals, int NRe, int NIm);

// perform Nit iterations of newton's method on a polynomial p
__global__ void newtonIterate(Complex *zVals, Polynomial P, Polynomial Pprime,
                                int NRe, int NIm, int Nit);

__host__ __device__ int findSolns(Complex *solns, Complex *zVals,
                                   int nSolns, int nVals);

__global__ void findClosestSoln(int *closest, Complex *zVals, int NRe, int NIm,
                                Complex *solns, int nSolns);

// L2 distance between two points
__host__ __device__ dfloat L2Distance(Complex z1, Complex z2);

__host__ __device__ dfloat L1Distance(Complex z1, Complex z2);

#endif
