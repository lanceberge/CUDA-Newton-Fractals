#include "point.h"
#include <math.h>

dfloat L2Distance(Point *z1, Point *z2)
{
    dfloat xDiff = z1->x - z2->x;
    dfloat yDiff = z1->y - z2->y;

    return sqrt((xDiff*xDiff) + (yDiff*yDiff));
}
