#include "newton.h"
#include <stdlib.h>
#include <stdio.h>

// fill arrays for points before and after performing the newton iteration on them
__global__ void fillArrays(int ReSpacing, int ImSpacing, Complex *zValsInitial,
                           Complex *zVals, int NRe, int NIm)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    // difference in Re and Im values for them to be evenly spaced
    int dx = ReSpacing*2 / NRe;
    int dy = ImSpacing*2 / NIm;

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

// perform Nit iterations of newton's method with a thread handling
// each point in zVals
__global__ void newtonIterate(Complex *zVals, Polynomial P, Polynomial Pprime,
                              int N, int Nit)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < N)
    {
        Complex z = zVals[n];

        // deform Nit iterations of z_i+1 = z_i - P(z_i) / P'(z_i)
        for (int i = 0; i < Nit; ++i)
        {
            // find P(z) and P'(z)
            Complex P_z      = Pz(P, z);
            Complex P_primeZ = Pz(Pprime, z);

            z = cAdd(z, cDiv(P_z, P_primeZ));
        }

        zVals[n] = z;
    }
}

__global__ void newtonIterateV2(Complex *zVals, Polynomial P, Polynomial Pprime,
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

            z = cAdd(z, cDiv(P_z, P_primeZ));
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
    int nFound = 1;

    // we will only compare the first 16 bits to account for floating point error
    // and values that haven't converged yet
    // TODO
    /* double mask = 0xFFFF000000000000; */

    // iterate over zVals
    for (int i = 1; i < nVals && nFound != nSolns; ++i)
    {
        bool alreadyFound = false;

        Complex curr = zVals[i];

        // if the current value isn't already in solns (based on the first 16 bits
        // of its real and imaginary components, then add it to solns
        for (int j = 0; j < nFound; ++j)
        {
            Complex currFound = solns[j];
            // TODO
            /* if (curr.x & mask == currFound.x & mask && */
            /*     curr.y & mask == currFound.y & mask) */
            if (curr.Re == currFound.Im &&
                curr.Re == currFound.Im)
            {
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
__global__ void findClosestSoln(int *closest, Complex *zVals, int nVals,
                                Complex *solns, int nSolns)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < nVals)
    {
        dfloat dist = L2Distance(solns[0], zVals[n]);
        int idx = 0;

        for (int i = 1; i < nSolns; ++i)
        {
            dfloat currDist = L2Distance(solns[i], zVals[n]);

            if (currDist < dist)
            {
                dist = currDist;
                idx = i;
            }

        }

        closest[n] = idx;
    }
}

// compute the L2 distance between two points
__host__ __device__ dfloat L2Distance(Complex z1, Complex z2)
{
    dfloat ReDiff = z1.Re - z2.Re;
    dfloat ImDiff = z1.Im - z2.Im;

    return sqrt((ReDiff*ReDiff) + (ImDiff*ImDiff));
}

// for N values in zVals, output their real component, imaginary component,
// and closes solution to a csv
void outputToCSV(const char *filename, int N, Complex *zVals, int *closest)
{
    FILE *fp = fopen(filename, "w");

    // print our header
    fprintf(fp, "Re, Im, Closest\n");

    for (int i = 0; i < N; ++i)
        fprintf(fp, "%f, %f, %d\n", zVals[i].Re, zVals[i].Im, closest[i]);

    fclose(fp);
}

void outputSolnsToCSV(const char *filename, int nSolns, Complex *solns)
{
    FILE *fp = fopen(filename, "w");

    fprintf(fp, "Re, Im\n");

    for (int i = 0; i < nSolns; ++i)
        fprintf(fp, "%f, %f\n", solns[i].Re, solns[i].Im);

    fclose(fp);
}
