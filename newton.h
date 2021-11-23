#ifndef __NEWTON_H__
#define __NEWTON_H__

#include "cuda.h"
#include "polynomial.h"
#include <complex>

// fill arrays for points before and after performing the newton iteration on them
__global__ void fillArrays(int ReSpacing, int ImSpacing, std::complex<dfloat> *zValsInitial,
                std::complex<dfloat> *zVals, int NRe, int NIm);

// perform Nit iterations of newton's method on a polynomial p
__global__ void newtonIterate(std::complex<dfloat> *zVals, Polynomial *P, Polynomial *Pprime,
                              int N, int Nit);

__host__ __device__ void findSolns(std::complex<dfloat> *solns, std::complex<dfloat> *zVals,
                                   int nSolns, int nVals);
// L2 distance between two points
__host__ __device__ dfloat L2Distance(std::complex<dfloat> z1, std::complex<dfloat> z2);

void outputToCSV(const char *filename, int N, std::complex<dfloat> *zVals, int *closest);

#endif
