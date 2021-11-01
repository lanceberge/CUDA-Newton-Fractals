#ifndef __NEWTON_H__
#define __NEWTON_H__

#include "cuda.h"

#define dfloat float

typedef struct Point
{
    dfloat x;
    dfloat y;
} Point;

typedef struct PointChange
{
    Point *before;
    Point *after;
} Change;

dfloat L2Distance(Point p1, Point p2);

#endif
