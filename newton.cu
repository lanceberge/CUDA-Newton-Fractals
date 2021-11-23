#include "newton.h"
#include <math.h>

// fill arrays for points before and after performing the newton iteration on them
__global__ void fillArrays(int ReSpacing, int ImSpacing, std::complex<dfloat> *zValsInitial,
                           std::complex<dfloat> *zVals, int NRe, int NIm)
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
        zValsInitial[x + NRe*y] = std::complex<dfloat> (Re, Im);
        zVals       [x + NRe*y] = std::complex<dfloat> (Re, Im);
    }
}

// perform Nit iterations of newton's method with a thread handling
// each point in zVals
__global__ void newtonIterate(std::complex<dfloat> *zVals, Polynomial *P, Polynomial *Pprime,
                              int N, int Nit)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < N)
    {
        std::complex<dfloat> z = zVals[n];

        // deform Nit iterations of z_i+1 = z_i - P(z_i) / P'(z_i)
        for (int i = 0; i < Nit; ++i)
        {
            // find P(z) and P'(z)
            std::complex<dfloat> P_z = Pz(P, z);
            std::complex<dfloat> P_primeZ = Pz(Pprime, z);

            // TODO
            z = z + P_z/P_primeZ;
        }

        zVals[n] = z;
    }
}

// for each solution in zVals, find the solution it's closest to based on L1 distance
__global__ void findClosestSoln(int *closest, std::complex<dfloat> *zVals, int nVals,
                                std::complex<dfloat> *solns, int nSolns)
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
__host__ __device__ void findSolns(std::complex<dfloat> *solns, std::complex<dfloat> *zVals,
                                   int nSolns, int nVals)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < nVals)
    {
        int nFound = 1;

        // we will only compare the first 16 bits to account for floating point error
        // and values that haven't converged yet
        float mask = 0xFFFF0000;

        // iterate over zVals
        for (int i = 1; i < nVals && nFound != n; ++i)
        {
            bool alreadyFound = false;

            std::complex<dfloat> curr = zVals[i];

            // if the current value isn't already in solns (based on the first 16 bits
            // of its real and imaginary components, then add it to solns
            for (int j = 0; j < nFound; ++j)
            {
                std::complex<dfloat> currFound = solns[j];
                // TODO
                /* if (curr.real() & mask == currFound.real() & mask && */
                /*     curr.imag() & mask == currFound.imag() & mask) */
                if (curr.real() == currFound.real() &&
                    curr.imag() == currFound.imag())
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
    }
}

// compute the L2 distance between two points
__host__ __device__ dfloat L2Distance(std::complex<dfloat> z1, std::complex<dfloat> z2)
{
    // TODO
    dfloat ReDiff = z1.real() - z2.real();
    dfloat ImDiff = z1.imag() - z2.imag();

    return sqrt((ReDiff*ReDiff) + (ImDiff*ImDiff));
}

// for N values in zVals, output their real component, imaginary component,
// and closes solution to a csv
void outputToCSV(const char *filename, int N, std::complex<dfloat> *zVals, int *closest)
{
    FILE *fp = fopen(filename, "w");

    // print our header
    fprintf(fp, "Re, Im, Closest");

    for (int i = 0; i < N; ++i)
        // TODO
        fprintf(fp, "%f, %f, %d", zVals[i].real(), zVals[i].imag(), closest[i]);

    fclose(fp);
}
