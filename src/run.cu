#include "newton.h"
#include <string>

static int NRe;
static int NIm;

void performIteration(Polynomial P, int ReSpacing, int ImSpacing,
                      std::string filename, int Nits);

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

    NRe        = atoi(argv[1]);
    NIm        = atoi(argv[2]);
    char *test = argv[3];

    Polynomial P;

    // test on -4x^3 + 6x^2 + 2x = 0, which has roots
    // 0, ~1.78, ~-.28
    if (strcmp(test, "smallTest") == 0)
    {
        int order = 3;

        // create a polynomial
        dfloat *coeffs = new dfloat[4] {-4, 6, 2, 0};
        P.coeffs = coeffs;
        P.order = order;

        // the spacing on our grid, i.e. 1000 => run iteration on Nx and Ny evenly
        // spaced points from -1000 to 1000 on x and y
        dfloat ReSpacing = 4;
        dfloat ImSpacing = 4;

        int Nits = 100;

        performIteration(P, ReSpacing, ImSpacing, "smallTest", Nits);
    }

    else if (strcmp(test, "bigTest") == 0)
    {
        // create a random order 7 polynomial
        srand48(123456);

        int order = 7;
        dfloat *coeffs = (dfloat *)malloc((order + 1)*sizeof(dfloat));

        for (int i = 0; i < order + 1; ++i)
        {
            coeffs[i] = -10 + 20*(drand48());
        }

        P.coeffs = coeffs;
        P.order = order;

        dfloat ReSpacing = 4;
        dfloat ImSpacing = 4;

        int Nits = 100;

        performIteration(P, ReSpacing, ImSpacing, "bigTest", Nits);
    }

    return 0;
}

void performIteration(Polynomial P, int ReSpacing, int ImSpacing,
                      std::string filename, int Nits)
{
    // total number of points
    int N = NRe*NIm;

    // device arrays
    Complex *zValsInitial;
    Complex *zVals;
    Complex *solns;
    int *closest;

    dim3 B(16, 16, 1);
    dim3 G((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);

    // P' - derivative of P
    Polynomial Pprime = derivative(P);

    // device versions for newtonIterate
    Polynomial c_P      = deviceP(P);
    Polynomial c_Pprime = deviceP(Pprime);

    // arrays for solutions
    int order = P.order;

    // arrays for initial points and points following iteration
    cudaMalloc(&zValsInitial, N*sizeof(Complex));
    cudaMalloc(&zVals,        N*sizeof(Complex));

    fillArrays <<< G, B >>> (ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);

    // then perform the newton iteration and copy result back to host
    newtonIterate <<< G, B >>> (zVals, c_P, c_Pprime, NRe, NIm, Nits);

    Complex *h_zValsInitial = (Complex *)malloc(N*sizeof(Complex));
    Complex *h_zVals        = (Complex *)malloc(N*sizeof(Complex));
    cudaMemcpy(h_zValsInitial, zValsInitial, N*sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zVals,        zVals,        N*sizeof(Complex), cudaMemcpyDeviceToHost);

    // find the solutions to this polynomial - the unique points in zVals
    cudaMalloc(&solns, order*sizeof(Complex));
    Complex *h_solns = (Complex *)malloc(order * sizeof(Complex));
    int nSolns = findSolns(h_solns, h_zVals, order, N);

    // copy to device
    cudaMemcpy(solns, h_solns, nSolns*sizeof(Complex), cudaMemcpyHostToDevice);

    // fill *closest with an integer corresponding to the solution its closest to
    // i.e. 0 for if this point is closest to solns[0]
    int *h_closest = (int *)malloc(N * sizeof(int));
    cudaMalloc(&closest, N*sizeof(int));

    findClosestSoln <<< G, B >>> (closest, zVals, NRe, NIm, solns, nSolns);

    // copy results back to host
    cudaMemcpy(h_closest, closest, N*sizeof(int), cudaMemcpyDeviceToHost);

    // output data and solutions to CSVs
    std::string outputFilename = "data/"+filename+"Data.csv";
    std::string solnFilename   = "data/"+filename+"Solns.csv";

    outputToCSV(outputFilename.c_str(), N, h_zValsInitial, h_closest);
    outputSolnsToCSV(solnFilename.c_str(), nSolns, h_solns);

    // free memory
    cudaFree(zVals)          ; free(h_zVals)       ;
    cudaFree(zValsInitial)   ; free(h_zValsInitial);
    cudaFree(c_P.coeffs)     ; free(P.coeffs)      ;
    cudaFree(c_Pprime.coeffs); free(Pprime.coeffs) ;
    cudaFree(closest)        ; free(h_closest)     ;
    cudaFree(solns)          ; free(h_solns)       ;
}
