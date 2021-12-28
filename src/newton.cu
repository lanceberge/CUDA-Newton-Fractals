#include "newton.h"

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

// perform the iteration on each val in zVals
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
            Complex P_z      = P.c_Pz(z);
            Complex P_primeZ = Pprime.c_Pz(z);

            // perform one iteration
            z = z - P_z/P_primeZ;
        }

        zVals[x + NRe*y] = z;
    }
}

// zVals - the complex values following running Newton's iteration
// nSolns - order of the polynomial
// after running the iteration, zVals should represent n unique values corresponding
// to the solutions of the polynomial, this function finds those unique values
int findSolns(const Polynomial& P, Complex *solns, Complex *zVals,
                                   int nSolns, int nVals)
{
    int nFound = 0;

    // iterate over zVals
    for (int i = 0; i < nVals && nFound != nSolns; ++i)
    {
        bool alreadyFound = false;

        Complex curr = zVals[i];

        // if this isn't a valid solution, i.e. the iteration didn't converge
        // to a solution for that initial guess
        Complex P_z = P.h_Pz(curr);

        // if this value isn't a root; if P(z)'s Re or Im value's aren't 0
        if (!(fabs(P_z.Re) < 1e-10 && fabs(P_z.Im) < 1e-10))
            continue;

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
                                Complex *solns, int nSolns, int norm)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < NRe && y < NIm)
    {
        Complex z = zVals[x + NRe*y];
        dfloat dist;

        if (norm == 1)
            dist = L1Distance(solns[0], z);

        else
            dist = L2Distance(solns[0], z);

        int idx = 0;

        for (int i = 1; i < nSolns; ++i)
        {
            dfloat currDist;

            if (norm == 1)
                currDist = L1Distance(solns[i], z);

            else
                currDist = L2Distance(solns[i], z);

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
__device__ dfloat L2Distance(const Complex& z1, const Complex& z2)
{
    dfloat ReDiff = z1.Re - z2.Re;
    dfloat ImDiff = z1.Im - z2.Im;

    return sqrt((ReDiff*ReDiff) + (ImDiff*ImDiff));
}

// compute the L1 distance between two points
__device__ dfloat L1Distance(const Complex& z1, const Complex& z2)
{
    dfloat ReDiff = z1.Re - z2.Re;
    dfloat ImDiff = z1.Im - z2.Im;

    return fabs(ReDiff) + fabs(ImDiff);
}
