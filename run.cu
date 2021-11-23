#include "newton.h"

int main(int argc, int **argv)
{
    if (argc < 4)
    {
        printf("Usage: ./newton {NRe} {NIm} {Test}\n");
        printf("NRe - Number of real points to run iteration on\n");
        printf("NIm - number of imaginary points to run iteration on\n");
        // printf("Nit - Number of iterations to run\n");
        printf("Test - Which test to run\n");
        exit(-1);
    }

    int NRe = atoi(argv[2]);
    int NIm = atoi(argv[3]);
    // int Nit = atoi(argv[4]);
    char *test = argv[4];

    dfloat complex *zValsInitial, zVals;
    dfloat complex *h_zValsInitial, h_zVals;
    Polynomial *P;
    Polynomial *Pprime;
    dfloat complex *solns;
    int *closest;

    if (strcmp(test, "smallTest") == 0)
    {
        // the spacing on our grid, i.e. 1000 => run iteration on Nx and Ny evenly
        // spaced points from -1000 to 1000 on x and y
        int ReSpacing = 1000;
        int ImSpacing = 500;

        // total number of points
        int N = NRe*NIm;

        // arrays for initial points and points following iteration
        cudaMalloc(&zValsInitial, N*sizeof(dfloat complex));
        cudaMalloc(&zVals,        N*sizeof(dfloat complex));

        h_zValsInitial = (dfloat complex *)malloc(NRe*NIm*sizeof(dfloat complex));
        h_zVals        = (dfloat complex *)malloc(NRe*NIm*sizeof(dfloat complex));

        // create a polynomial
        int order = 4;

        P = (Polynomial *)malloc(sizeof(Polynomial *));
        P->order = order;
        P->coeffs = {1, 2, 3, 4, 5};

        Pprime = derivative(P);

        int B = 256;
        int G = N + B - 1 / B;

        dim3 B2(16, 16, 1);
        dim3 G2((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);

        fillArrays    <<< G2, B2 >>> (ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);
        newtonIterate <<< G, B>>>    (zVals, P, Pprime, N, Nit);

        cudaMemcpy(h_zValsInitial, zValsInitial, N*sizeof(dfloat complex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_zVals, zVals,               N*sizeof(dfloat complex), cudaMemcpyDeviceToHost);

        solns = (dfloat complex *)malloc(4 * sizeof(dfloat complex));

        // find the solutions to this polynomial
        findSolns(solns, h_zVals, 4, N);

        closest = (dfloat complex *)malloc(N * sizeof(dfloat complex));
        outputToCSV("smallTest.csv", N, h_zVals, closest);
    }

    // free memory
    cudaFree(zVals); cudaFree(zValsInitial);
    free(h_zVals); free(h_zValsInitial);
    freePolynomial(P); freePolynomial(Pprime);
    free(solns); free(closest);

    return 0;
}
