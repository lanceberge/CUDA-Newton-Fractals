#ifndef __POINT_H__
#define __POINT_H__

#define dfloat float

// an xy point
typedef struct Point
{
    dfloat x;
    dfloat y;
} Point;

// L2 distance between two points
dfloat L2Distance(Point *z1, Point *z2);

#endif
