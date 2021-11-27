#include "newton.h"
#include <string>
#include <stdlib.h>

static int NRe;
static int NIm;
static int ReSpacing;
static int ImSpacing;

void iterate(Polynomial c_P, Polynomial c_Pprime, int Nits, Complex *zVals, Complex *h_zVals);

void outputSolns(Complex *h_zVals, Complex *h_zValsInitial,
                 Complex **h_solns, int nSolns, std::string filename);

void outputVals(Complex *zVals, Complex *h_zVals, Complex *h_solns, Complex *h_zValsInitial,
                int nSolns, std::string filename, int step=-1);

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: ./newton <NRe> <NIm> <Test> [step]\n");
        printf("NRe  - Number of real points to run iteration on\n");
        printf("NIm  - number of imaginary points to run iteration on\n");
        printf("Test - Which test to run\n");
        printf("Step - optional, use to output at each step\n");
        exit(-1);
    }

    NRe        = atoi(argv[1]);
    NIm        = atoi(argv[2]);
    char *test = argv[3];

    int N = NRe*NIm;

    Polynomial P;

    Complex *zValsInitial;
    Complex *zVals;
    int order;

    // arrays for initial points and points following iteration
    cudaMalloc(&zValsInitial, N*sizeof(Complex));
    cudaMalloc(&zVals,        N*sizeof(Complex));

    Complex *h_zValsInitial = (Complex *)malloc(N*sizeof(Complex));
    Complex *h_zVals        = (Complex *)malloc(N*sizeof(Complex));

    dim3 B(16, 16, 1);
    dim3 G((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);

    // test on -4x^3 + 6x^2 + 2x = 0, which has roots
    // 0, ~1.78, ~-.28
    if (strcmp(test, "smallTest") == 0)
    {
        order = 3;

        // create a polynomial
        dfloat *coeffs = new dfloat[4] {-4, 6, 2, 0};
        P.coeffs = coeffs;
        P.order = order;

        // the spacing on our grid, i.e. 1000 => run iteration on Nx and Ny evenly
        // spaced points from -1000 to 1000 on x and y
        ReSpacing = 4;
        ImSpacing = 4;
    }

    else if (strcmp(test, "bigTest") == 0)
    {
        // create a random order 7 polynomial
        srand48(123456);

        order = 7;
        dfloat *coeffs = (dfloat *)malloc((order + 1)*sizeof(dfloat));

        for (int i = 0; i < order + 1; ++i)
        {
            coeffs[i] = -10 + 20*(drand48());
        }

        P.coeffs = coeffs;
        P.order = order;

        ReSpacing = 4;
        ImSpacing = 4;
    }

    else
    {
        return 0;
    }

    // P' - derivative of P
    Polynomial Pprime = derivative(P);

    // device versions for newtonIterate
    Polynomial c_P      = deviceP(P);
    Polynomial c_Pprime = deviceP(Pprime);

    Complex *h_solns = (Complex *)malloc(order*sizeof(Complex));

    fillArrays <<< G, B >>> (ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);

    cudaMemcpy(h_zValsInitial, zValsInitial, N*sizeof(Complex), cudaMemcpyDeviceToHost);


    // perform 100 iterations then output solutions
    iterate(c_P, c_Pprime, 100, zVals, h_zVals);

    // output solutions to file and store them
    outputSolns(h_zVals, h_zValsInitial, &h_solns, order, test);
    outputVals(zVals, h_zVals, h_solns, h_zValsInitial, order, test);

    if (argc >= 5 && strcmp(argv[4], "step") == 0)
    {
        // reset arrays
        fillArrays <<< G, B >>> (ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);

        cudaMemcpy(h_zVals, zVals, N*sizeof(Complex), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 100; ++i)
        {
            // output then perform 1 iteration
            outputVals(zVals, h_zVals, h_solns, h_zValsInitial, order, test, i);
            iterate(c_P, c_Pprime, 1, zVals, h_zVals);
        }
    }


    cudaFree(zVals)          ; free(h_zVals)       ;
    cudaFree(zValsInitial)   ; free(h_zValsInitial);
    cudaFree(c_P.coeffs)     ; free(P.coeffs)      ;
    cudaFree(c_Pprime.coeffs);

    free(h_solns);
    return 0;
}

void iterate(Polynomial c_P, Polynomial c_Pprime, int Nits, Complex *zVals, Complex *h_zVals)
{
    dim3 B(16, 16, 1);
    dim3 G((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);

    // then perform the newton iteration and copy result back to host
    newtonIterate <<< G, B >>> (zVals, c_P, c_Pprime, NRe, NIm, Nits);

    // copy result to host
    cudaMemcpy(h_zVals, zVals, NRe*NIm*sizeof(Complex), cudaMemcpyDeviceToHost);
}

void outputSolns(Complex *h_zVals, Complex *h_zValsInitial,
                 Complex **h_solns, int nSolns, std::string filename)
{
    // total number of points
    // find the solutions to this polynomial - the unique points in zVals
    *h_solns = (Complex *)malloc(nSolns * sizeof(Complex));
    nSolns = findSolns(*h_solns, h_zVals, nSolns, NRe*NIm);

    std::string solnFilename   = "data/"+filename+"Solns.csv";

    outputSolnsToCSV(solnFilename.c_str(), nSolns, *h_solns);
}

void outputVals(Complex *zVals, Complex *h_zVals, Complex *h_solns, Complex *h_zValsInitial,
                int nSolns, std::string filename, int step)
{
    dim3 B(16, 16, 1);
    dim3 G((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);

    int *closest;
    cudaMalloc(&closest, NRe*NIm*sizeof(int));

    Complex *solns;
    cudaMalloc(&solns, nSolns*sizeof(Complex));
    cudaMemcpy(solns, h_solns, nSolns*sizeof(Complex), cudaMemcpyHostToDevice);

    findClosestSoln <<< G, B >>> (closest, zVals, NRe, NIm, solns, nSolns);

    // fill *closest with an integer corresponding to the solution its closest to
    // i.e. 0 for if this point is closest to solns[0]
    int *h_closest = (int *)malloc(NRe*NIm * sizeof(int));

    // copy results back to host
    cudaMemcpy(h_closest, closest, NRe*NIm*sizeof(int), cudaMemcpyDeviceToHost);

    // output data and solutions to CSVs
    std::string outputFilename;
    if (step == -1)
        outputFilename = "data/"+filename+"Data.csv";

    else
        outputFilename = "data/"+filename+"Data-"+std::to_string(step)+".csv";


    outputToCSV(outputFilename.c_str(), NRe*NIm, h_zValsInitial, h_closest);

    cudaFree(closest); free(h_closest);
    cudaFree(solns);
}
