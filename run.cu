#include "cuda.h"
#include "newton.h"

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

    int NRe = atoi(argv[1]);
    int NIm = atoi(argv[2]);
    char *test = argv[3];

    std::complex<dfloat> *zValsInitial, *zVals;
    std::complex<dfloat> *h_zValsInitial, *h_zVals;

    Polynomial *P;
    Polynomial *Pprime;
    std::complex<dfloat> *solns;
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
        cudaMalloc(&zValsInitial, N*sizeof(std::complex<dfloat>));
        cudaMalloc(&zVals,        N*sizeof(std::complex<dfloat>));

        h_zValsInitial = (std::complex<dfloat> *)malloc(NRe*NIm*sizeof(std::complex<dfloat>));
        h_zVals        = (std::complex<dfloat> *)malloc(NRe*NIm*sizeof(std::complex<dfloat>));

        // create a polynomial
        int order = 4;

        P = (Polynomial *)malloc(sizeof(Polynomial *));
        P->order = order;
        int coeffs[5] = {1, 2, 3, 4, 5};
        P->coeffs = coeffs;

        Pprime = derivative(P);

        int B = 256;
        int G = N + B - 1 / B;

        dim3 B2(16, 16, 1);
        dim3 G2((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);

        fillArrays    <<< G2, B2 >>> (ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);
        newtonIterate <<< G, B>>>    (zVals, P, Pprime, N, 100);

        cudaMemcpy(h_zValsInitial, zValsInitial, N*sizeof(std::complex<dfloat>), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_zVals, zVals,               N*sizeof(std::complex<dfloat>), cudaMemcpyDeviceToHost);

        solns = (std::complex<dfloat> *)malloc(4 * sizeof(std::complex<dfloat>));

        // find the solutions to this polynomial
        findSolns(solns, h_zVals, 4, N);

        closest = (int *)malloc(N * sizeof(int));
    }

    // free memory
    cudaFree(zVals); cudaFree(zValsInitial);
    free(h_zVals); free(h_zValsInitial);
    freePolynomial(P); freePolynomial(Pprime);
    free(solns); free(closest);

    return 0;
}
