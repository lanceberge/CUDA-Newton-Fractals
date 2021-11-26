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

    // the spacing on our grid, i.e. 1000 => run iteration on Nx and Ny evenly
    // spaced points from -1000 to 1000 on x and y
    int ReSpacing = 1000;
    int ImSpacing = 500;

    // total number of points
    int N = NRe*NIm;

    Complex *zValsInitial, *h_zValsInitial;
    Complex *zVals,        *h_zVals;
    Complex *solns,        *h_solns;

    Polynomial P;
    Polynomial Pprime;
    Polynomial c_P;
    Polynomial c_Pprime;

    int *closest, *h_closest;

    dim3 B(16, 16, 1);
    dim3 G((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);

    // arrays for initial points and points following iteration
    cudaMalloc(&zValsInitial, N*sizeof(Complex));
    cudaMalloc(&zVals,        N*sizeof(Complex));

    // host arrays for points
    h_zValsInitial = (Complex *)malloc(N*sizeof(Complex));
    h_zVals        = (Complex *)malloc(N*sizeof(Complex));

    h_closest = (int *)malloc(N * sizeof(int));
    cudaMalloc(&closest, N*sizeof(int));

    // test on -4x^3 + 6x^2 + 2x = 0, which has roots
    // 0, ~1.78, ~-.28
    if (strcmp(test, "smallTest") == 0)
    {
        int order = 3;

        // arrays for solutions
        cudaMalloc(&solns, order*sizeof(Complex));
        h_solns = (Complex *)malloc(order * sizeof(Complex));

        // create a polynomial
        P.order = order;
        /* dfloat coeffs[4] = {-4, 6, 2, 0}; */
        dfloat *coeffs = new dfloat[4] {-4, 6, 2, 0};
        P.coeffs = coeffs;

        // it's derivative
        Pprime = derivative(P);

        // device versions for newtonIterate
        Polynomial c_P      = deviceP(P);
        Polynomial c_Pprime = deviceP(Pprime);

        // fill our arrays with points
        fillArrays <<< G, B >>> (ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);

        // then perform the newton iteration
        newtonIterate <<< G, B >>> (zVals, c_P, c_Pprime, NRe, NIm, 100);

        // copy results back to host
        cudaMemcpy(h_zValsInitial, zValsInitial, N*sizeof(Complex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_zVals,        zVals,        N*sizeof(Complex), cudaMemcpyDeviceToHost);

        // find the solutions to this polynomial - the unique points in zVals
        int nSolns = findSolns(h_solns, h_zVals, order, N);

        // copy to device
        cudaMemcpy(solns, h_solns, nSolns*sizeof(Complex), cudaMemcpyHostToDevice);

        // fill *closest with an integer corresponding to the solution its closest to
        // i.e. 0 for if this point is closest to solns[0]
        findClosestSoln <<< G, B >>> (closest, zVals, NRe, NIm, solns, nSolns);

        // copy results back to host
        cudaMemcpy(h_closest, closest, N*sizeof(int), cudaMemcpyDeviceToHost);

        // output data and solutions to csvs
        outputToCSV("data/smallData.csv", N, h_zValsInitial, h_closest);
        outputSolnsToCSV("data/smallSolns.csv", nSolns, h_solns);
    }

    // free memory
    cudaFree(zVals)          ; free(h_zVals)           ;
    cudaFree(zValsInitial)   ; free(h_zValsInitial);
    cudaFree(c_P.coeffs)     ; free(P.coeffs)          ;
    cudaFree(c_Pprime.coeffs); free(Pprime.coeffs)     ;
    cudaFree(closest)        ; free(h_closest)         ;
    cudaFree(solns)          ; free(h_solns)           ;

    return 0;
}
