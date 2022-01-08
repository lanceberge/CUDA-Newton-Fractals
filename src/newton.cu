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

    if (x < NRe && y < NIm) {
        // Real value here - evenly spaced from -ReSpacing to Respacing
        // with NRe elements, same for Im
        dfloat Re = x*dx - ReSpacing;
        dfloat Im = y*dy - ImSpacing;

        // fill zVals arrays in row-major format
        zValsInitial[x + NRe*y] = Complex(Re, Im);
        zVals       [x + NRe*y] = Complex(Re, Im);
    }
}

// perform the iteration on each val in zVals
__global__ void newtonIterate(Complex *zVals, Polynomial P, Polynomial Pprime,
                                int NRe, int NIm, int Nit)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < NRe && y < NIm) {
        Complex z = zVals[x + NRe*y];

        // perform Nit iterations of z_i+1 = z_i - P(z_i) / P'(z_i)
        for (int i = 0; i < Nit; ++i) {
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
    for (int i = 0; i < nVals && nFound != nSolns; ++i) {
        bool alreadyFound = false;

        Complex curr = zVals[i];

        // if this isn't a valid solution, i.e. the iteration didn't converge
        // to a solution for that initial guess
        Complex P_z = P.h_Pz(curr);

        // if this value isn't a root; if P(z)'s Re or Im value's aren't 0
        if (P_z != Complex(0, 0))
            continue;

        // if the current value isn't already in solns, then add it to solns
        for (int j = 0; j < nFound; ++j) {
            Complex currFound = solns[j];

            if (curr == currFound) {
                alreadyFound = true;
                break;
            }
        }

        // if this solution isn't already in solutions, add it
        if (!alreadyFound) {
            solns[nFound] = curr;
            ++nFound;
        }
    }

    return nFound;
}

__global__ void deviceFindSolns(Polynomial P, Complex *solns, Complex *zVals,
                                int nVals)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    // if the index is valid and the value there is a solution
    if (x < nVals) {
        Complex zVal = zVals[x];

        // if this value isn't a root
        if (P.c_Pz(zVal) != Complex(0, 0)) {
            return;
        }

        // convert our real and imaginary components to integers - intentionally losing
        // precision to make sure if a solution is approximately converged, it doesn't hash
        // to a different spot (e.g. we want 0.85 and 0.850001 to hash to the same spot)
        unsigned int Re = (int)(zVal.Re * 1000);
        unsigned int Im = (int)(zVal.Im * 1000);

        // hash these values to an index - this method is a good way to hash
        // 2d values according to:
        // https://stackoverflow.com/questions/2634690/good-hash-function-for-a-2d-index
        unsigned int idx = ((53 + hash(Re)) * 53 + hash(Im));


        // linear probe and insert where possible
        for (int i = 0; i <= P.order; ++i) {
            unsigned int probeIdx = (idx + i) % P.order;

            // if this value is already in the table
            __syncthreads();
            if (solns[probeIdx] == zVal || solns[idx % P.order] == zVal) {
                return;
            }

            __syncthreads();
            if (solns[probeIdx] == Complex(0, 0)) {
                solns[probeIdx] = zVal;
                return;
            }
        }
    }
}

// for each solution in zVals, find the solution it's closest to based on L1 distance
__global__ void findClosestSoln(int *closest, Complex *zVals, int NRe, int NIm,
                                Complex *solns, Polynomial P, int norm)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < NRe && y < NIm) {

        Complex z = zVals[x + NRe*y];
        dfloat dist;

        if (norm == 1)
            dist = L1Distance(solns[0], z);

        else
            dist = L2Distance(solns[0], z);

        int idx = 0;

        for (int i = 1; i < P.order; ++i) {
            Complex root(0, 0);

            // the default value written into solns by cudaMalloc is the
            // 0,0 complex, which may or may not actually be a solution
            // thus if the value in solns is 0,0, we verify that it is a root
#ifdef deviceFindSolns // TODO
            if (solns[i] == root && P.c_Pz(solns[i]) == root) {
#endif
                dfloat currDist;

                if (norm == 1)
                    currDist = L1Distance(solns[i], z);

                else
                    currDist = L2Distance(solns[i], z);

                if (currDist < dist) {
                    dist = currDist;
                    idx = i;
                }
#ifdef deviceFindSolns
            }
#endif
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

// from: https://stackoverflow.com/questions/664014/what-
// integer-hash-function-are-good-that-accepts-an-integer-hash-key
__device__ unsigned int hash(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}
