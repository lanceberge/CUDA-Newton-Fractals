#include "newton.h"
#include <math.h>

// perform an iteration of newton's method with a thread handling
// each point in points
__global__ void newtonIterate(PointChange **points, Polynomial *P, Polynomial *Pprime, int N, int Nit)
{
    int n = threadIdx.x + blockIdx.x + blockDim.x;

    if (n < N)
    {
        // peform Nit iterations of z_i+1 = z_i - P(z_i) / P'(z_i)
        for (int i = 0; i < Nit; ++i)
        {
            /* PointChange *p = points[i]; */
            Point *z = points[i]->after;

            // find P(z) and P'(z)
            Point Pz = Pz(P, z);
            Point Pprimez = Pz(Pprime, z);
        }
    }
}

// compute the L2 distance between two points
dfloat L2Distance(complex dfloat z1, complex dfloat z2)
{
    dfloat ReDiff = creal(z1) - creal(z2);
    dfloat ImDiff = cimag(z1) - cimag(z2);

    return sqrt((ReDiff*ReDiff) + (ImDiff*ImDiff));
}
