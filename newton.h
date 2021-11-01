#ifndef __NEWTON_H__
#define __NEWTON_H__

#define dfloat float

#include "cuda.h"
#include "point.h"

// perform Nit iterations of newton's method on a polynomial p
__global__ void newtonIterate(Point *Pz, Point *PprimeZ, int N, int Nit, PointChange **points);

// serial newton iteration
void serialNewtonIterate(Point *Pz, Point *PprimeZ, int N, int Nit, PointChange **points);

#endif
