#include "newton.h"

int main(int argc, int **argv)
{
    if (argc < 4)
    {
        printf("Usage: ./newton {Nx} {Ny} {Nit}\n");
        printf("NRe - Number of real points to run iteration on\n");
        printf("NIm - number of imaginary points to run iteration on\n");
        printf("Nit - Number of iterations to run\n");
        exit(-1);
    }

    int NRe = atoi(argv[2]);
    int NIm = atoi(argv[3]);
    int Nit = atoi(argv[4]);

    // the spacing on our grid, i.e. 1000 => run iteration on Nx and Ny evenly
    // spaced points from -1000 to 1000 on x and y
    int ReSpacing = 1000;
    int ImSpacing = 500;

    // total number of points
    int N = NRe*NIm;

    // arrays for initial points and points following iteration
    dfloat complex *zValsInitial, zVals;
    cudaMalloc(&zValsInitial, N*sizeof(dfloat complex));
    cudaMalloc(&zVals, N*sizeof(dfloat complex));

    /* dfloat complex zValsInitial= (dfloat complex *)malloc(Nx*Ny*sizeof(dfloat complex)); */
    /* dfloat complex *zVals = (dfloat complex *)malloc(Nx*Ny*sizeof(dfloat complex)); */

    int B = 256;
    int G = N + B - 1 / B;

    dim3 B2(16, 16, 1);
    dim3 G2((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);
    fillArrays <<< G2, B2 >>> (ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);

    // TODO find a good polynomial P and test on roots
    // TODO confirm output on host
    newtonIterate <<< G, B>>> (zVals, P, Pprime, N, Nit);
    return 0;
}
