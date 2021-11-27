#include "newton.h"
#include <stdlib.h>
#include <stdio.h>

// fill arrays for points before and after performing the newton iteration on them
__global__ void fillArrays(dfloat ReSpacing, dfloat ImSpacing, Complex *zValsInitial,
                           Complex *zVals, int NRe, int NIm)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    // difference in Re and Im values for them to be evenly spaced
    dfloat dx = ReSpacing*2 / NRe;
    dfloat dy = ImSpacing*2 / NIm;

    if (x < NRe && y < NIm)
    {
        // Real value here - evenly spaced from -ReSpacing to Respacing
        // with NRe elements, same for Im
        dfloat Re = x*dx - ReSpacing;
        dfloat Im = y*dy - ImSpacing;

        // fill zVals arrays in row-major format
        Complex zValInitial = {Re, Im};
        Complex zVal        = {Re, Im};

        zValsInitial[x + NRe*y] = zValInitial;
        zVals       [x + NRe*y] = zVal;
    }
}

__global__ void newtonIterate(Complex *zVals, Polynomial P, Polynomial Pprime,
                                int NRe, int NIm, int Nit)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < NRe && y < NIm)
    {
        Complex z = zVals[x + NRe*y];

        // perform Nit iterations of z_i+1 = z_i - P(z_i) / P'(z_i)
        for (int i = 0; i < Nit; ++i)
        {
            // find P(z) and P'(z)
            Complex P_z      = Pz(P, z);
            Complex P_primeZ = Pz(Pprime, z);

            z = cSub(z, cDiv(P_z, P_primeZ));
        }

        zVals[x + NRe*y] = z;
    }
}

// zVals - the complex values following running Newton's iteration
// nSolns - order of the polynomial
// after running the iteration, zVals should represent n unique values corresponding
// to the solutions of the polynomial, this function finds those unique values
__host__ __device__ int findSolns(Complex *solns, Complex *zVals,
                                   int nSolns, int nVals)
{
    int nFound = 0;

    // iterate over zVals
    for (int i = 0; i < nVals && nFound != nSolns; ++i)
    {
        bool alreadyFound = false;

        Complex curr = zVals[i];

        // if the current value isn't already in solns, then add it to solns
        for (int j = 0; j < nFound; ++j)
        {
            Complex currFound = solns[j];
            if (fabs(curr.Re - currFound.Re) < 1e-10 &&
                fabs(curr.Im - currFound.Im) < 1e-10) {

                alreadyFound = true;
                break;
            }
        }

        // if this solution isn't already in solutions, add it
        if (!alreadyFound)
        {
            solns[nFound] = curr;
            ++nFound;
        }
    }

    return nFound;
}

// for each solution in zVals, find the solution it's closest to based on L1 distance
__global__ void findClosestSoln(int *closest, Complex *zVals, int NRe, int NIm,
                                Complex *solns, int nSolns)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < NRe && y < NIm)
    {
        Complex z = zVals[x + NRe*y];
        dfloat dist = L2Distance(solns[0], z);
        int idx = 0;

        for (int i = 1; i < nSolns; ++i)
        {
            dfloat currDist = L2Distance(solns[i], z);

            if (currDist < dist)
            {
                dist = currDist;
                idx = i;
            }
        }

        closest[x + NRe*y] = idx;
    }
}

// compute the L2 distance between two points
__host__ __device__ dfloat L2Distance(Complex z1, Complex z2)
{
    dfloat ReDiff = z1.Re - z2.Re;
    dfloat ImDiff = z1.Im - z2.Im;

    return sqrt((ReDiff*ReDiff) + (ImDiff*ImDiff));
}

// compute the L1 distance between two points
__host__ __device__ dfloat L1Distance(Complex z1, Complex z2)
{
    dfloat ReDiff = z1.Re - z2.Re;
    dfloat ImDiff = z1.Im - z2.Im;

    return ReDiff + ImDiff;
}
