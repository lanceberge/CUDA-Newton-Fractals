#include "newton.h"
#include <math.h>

// fill arrays for points before and after performing the newton iteration on them
__global__ void fillArrays(int ReSpacing, int ImSpacing, dfloat complex *ZvalsInitial,
                           dfloat complex *zVals, int NRe, int NIm)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdy.y;

    // the starting value to fill real and imaginary values in our complex plane
    // i.e. if startRe is 1000, we will have Re values evenly spaced from -1000 to 1000
    int startRe = 0 - ReSpacing;
    int startIm = 0 - ImSpacing;

    // difference in Re and Im values for them to be evenly spaced
    int dx = ReSpacing*2 / Nx;
    int dy = ImSpacing*2 / Ny;

    if (x < NRe && y < NIm)
    {
        // Real value here - evenly spaced from -ReSpacing to Respacing
        // with NRe elements, same for Im
        dfloat Re = x*dx - ReSpacing;
        dfloat Im = y*dy - ImSpacing;

        // fill zVals arrays in row-major format
        zvalsInitial[x + NRe*y] = Re + I*Im;
        zvals       [x + NRe*y] = Re + I*Im;
    }
}

// perform Nit iterations of newton's method with a thread handling
// each point in zVals
__global__ void newtonIterate(dfloat complex *zVals, Polynomial *P, Polynomial *Pprime,
                              int N, int Nit)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < N)
    {
        dfloat complex z = zVals[n];

        // deform Nit iterations of z_i+1 = z_i - P(z_i) / P'(z_i)
        for (int i = 0; i < Nit; ++i)
        {
            // find P(z) and P'(z)
            dfloat complex Pz = Pz(P, z);
            dfloat complex Pprimez = Pz(Pprime, z);

            z = z + Pz/PprimeZ;
        }

        zVals[n] = z;
    }
}

// for each solution in zVals, find the solution it's closest to based on L1 distance
__global__ void findClosestSoln(int *closest; dfloat complex *zVals, int nVals, dfloat complex *solns, int nSolns)
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

// zVals - the complex values following running Newton's iteration
// nSolns - order of the polynomial
// after running the iteration, zVals should represent n unique values corresponding
// to the solutions of the polynomial, this function finds those unique values
__host__ __device__ dfloat complex *findSolns(dfloat complex *solns, dfloat complex *zVals,
                                              int nSolns, int nVals)
{
    int nFound = 1;

    // we will only compare the first 16 bits to account for floating point error
    // and values that haven't converged yet
    float mask = 0xFFFF0000;


    // iterate over zVals
    for (int i = 1; i < nVals && nFound != n; ++i)
    {
        bool alreadyFound = false;

        dfloat complex curr = zVals[i];

        // if the current value isn't already in solns (based on the first 16 bits
        // of its real and imaginary components, then add it to solns
        for (int j = 0; j < nFound; ++j)
        {
            dfloat complex currFound = solns[j];
            if (creal(curr) & mask == creal(currFound) & mask &&
                cimag(curr) & mask == cimag(currFound) & mask)
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

    return solns;
}

// compute the L2 distance between two points
__host__ __device__ dfloat L2Distance(dfloat complex z1, dfloat complex z2)
{
    dfloat ReDiff = creal(z1) - creal(z2);
    dfloat ImDiff = cimag(z1) - cimag(z2);

    return sqrt((ReDiff*ReDiff) + (ImDiff*ImDiff));
}

// for N values in zVals, output their real component, imaginary component,
// and closes solution to a csv
void outputToCSV(const char *filename, int N, float *zVals, int *closest)
{
    FILE *fp = fopen(filename, "w");

    // print our header
    fprintf(fp, "Re, Im, Closest");

    for (int i = 0; i < N; ++i)
        fprintf(fp, "%f, %f, %d", creal(zVals[i]), cimag(zVals[i]), closest[i]);

    fclose(fp);
}
// TODO output to CSV / copy back to host and output to CSV
