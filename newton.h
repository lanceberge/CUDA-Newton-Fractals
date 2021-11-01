#ifndef __NEWTON_H__
#define __NEWTON_H__

#include "cuda.h"
#include "polynomial.h"

// fill arrays for points before and after performing the newton iteration on them
__global__ void fillArrays(complex dfloat *ZvalsInitial, complex dfloat *zVals);

// perform Nit iterations of newton's method on a polynomial p
__global__ void newtonIterate(PointChange **points, Point *Pz, Point *PprimeZ, int N, int Nit);

// serial newton iteration
void serialNewtonIterate(complex dfloat *zVals, int N, int Nit);

// L2 distance between two points
dfloat L2Distance(dfloat complex z1, dfloat complex z2);

#endif

#endif
