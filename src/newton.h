#pragma once

#include "polynomial.h"

// fill arrays for points before and after performing the newton iteration on them
__global__ void fillArrays(dfloat ReSpacing, dfloat ImSpacing, Complex *zValsInitial,
                              Complex *zVals, int NRe, int NIm);

// perform Nit iterations of newton's method on a polynomial p
__global__ void newtonIterate(Complex *zVals, Polynomial P, Polynomial Pprime,
                                  int NRe, int NIm, int Nit);

// find all of the unique values in zVals
int findSolns(const Polynomial& P, Complex *solns, Complex *zVals,
                                     int nSolns, int nVals);

// for each val in zVals, find the solution in solns it's closest to
__global__ void findClosestSoln(int *closest, Complex *zVals, int NRe, int NIm,
                                   Complex *solns, int nSolns, int norm);

// L2 distance between two points
__device__ dfloat L2Distance(const Complex& z1, const Complex& z2);

// L1 distance between two points
__device__ dfloat L1Distance(const Complex& z1, const Complex& z2);
