#include "newton.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: ./newton {NRe} {NIm} {Test}\n");
        printf("NRe - Number of real points to run iteration on\n");
        printf("NIm - number of imaginary points to run iteration on\n");
        printf("Test - Which test to run\n");
        exit(-1);
    }

    int NRe    = atoi(argv[1]);
    int NIm    = atoi(argv[2]);
    char *test = argv[3];

    Complex *zValsInitial,   *zVals;
    Complex *h_zValsInitial, *h_zVals;

    Polynomial P;
    Polynomial Pprime;
    Complex    *solns;
    int        *closest;
    Polynomial c_P;
    Polynomial c_Pprime;


    // test on -4x^3 + 6x^2 + 2x = 0, which has roots
    // 0, ~1.78, ~-.28
    if (strcmp(test, "smallTest") == 0)
    {
        // the spacing on our grid, i.e. 1000 => run iteration on Nx and Ny evenly
        // spaced points from -1000 to 1000 on x and y
        int ReSpacing = 1000;
        int ImSpacing = 500;

        // total number of points
        int N = NRe*NIm;

        // arrays for initial points and points following iteration
        cudaMalloc(&zValsInitial, N*sizeof(Complex));
        cudaMalloc(&zVals,        N*sizeof(Complex));

        h_zValsInitial = (Complex *)malloc(NRe*NIm*sizeof(Complex));
        h_zVals        = (Complex *)malloc(NRe*NIm*sizeof(Complex));

        // create a polynomial
        int order = 3;

        P.order = order;
        dfloat coeffs[4] = {-4, 6, 2, 0};
        P.coeffs = coeffs;

        Pprime = derivative(P);

        dfloat *c_Pcoeffs;
        dfloat *c_Pprimecoeffs;

        cudaMalloc(&c_Pcoeffs, (order+1)*sizeof(dfloat));
        cudaMalloc(&c_Pprimecoeffs, order*sizeof(dfloat));

        cudaMemcpy(c_Pcoeffs,      P.coeffs,     (order+1)*sizeof(dfloat), cudaMemcpyHostToDevice);
        cudaMemcpy(c_Pprimecoeffs, Pprime.coeffs, order*sizeof(dfloat),    cudaMemcpyHostToDevice);

        c_P.coeffs      = c_Pcoeffs;
        c_Pprime.coeffs = c_Pprimecoeffs;

        int B = 256;
        int G = N + B - 1 / B;

        dim3 B2(16, 16, 1);
        dim3 G2((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);

        fillArrays      <<< G2, B2 >>> (ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);
        newtonIterateV2 <<< G2, B2 >>> (zVals, c_P, c_Pprime, NRe, NIm, 1);

        cudaMemcpy(h_zValsInitial, zValsInitial, N*sizeof(Complex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_zVals,        zVals,        N*sizeof(Complex), cudaMemcpyDeviceToHost);

        /* solns = (Complex *)malloc(order * sizeof(Complex)); */

        // find the solutions to this polynomial
        /* int nSolns = findSolns(solns, h_zVals, order, N); */

        closest = (int *)malloc(N * sizeof(int));
        outputToCSV("data.csv", N, h_zVals, closest);
        /* outputSolnsToCSV("solns.csv", nSolns, solns); */
    }

    // free memory
    cudaFree(zVals)   ; cudaFree(zValsInitial) ;
    free(h_zVals)     ; free(h_zValsInitial)   ;
    cudaFree(c_P.coeffs); cudaFree(c_Pprime.coeffs);
    /* free(solns)    ; free(closest)         ; */

    return 0;
}
