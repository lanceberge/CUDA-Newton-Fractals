#include "newton.h"
#include <string>
#include <stdlib.h>

static int NRe;
static int NIm;

// perform Nits iterations and copy result back to host
void iterate(Polynomial c_P, Polynomial c_Pprime, int Nits, Complex *zVals, Complex *h_zVals)
{
    dim3 B(16, 16, 1);
    dim3 G((NRe + 16 - 1)/16, (NIm + 16 - 1)/16);

    // then perform the newton iteration and copy result back to host
    newtonIterate <<< G, B >>> (zVals, c_P, c_Pprime, NRe, NIm, Nits);

    // copy result to host
    cudaMemcpy(h_zVals, zVals, NRe*NIm*sizeof(Complex), cudaMemcpyDeviceToHost);
}

// find the solutions given h_zVals and output to CSV
void outputSolns(Polynomial P, Complex *h_zVals, Complex **h_solns, int nSolns, int N, std::string filename)
{
    // total number of points
    // find the solutions to this polynomial - the unique points in zVals
    *h_solns = (Complex *)malloc(nSolns * sizeof(Complex));
    nSolns = findSolns(P, *h_solns, h_zVals, nSolns, N);

    std::string solnFilename = "data/"+filename+"Solns.csv";

    FILE *fp = fopen(solnFilename.c_str(), "w");

    // print our header
    fprintf(fp, "Re, Im\n");

    Complex *solns = *h_solns;

    for (int i = 0; i < nSolns; ++i)
    {
        fprintf(fp, "%f, %f\n", solns[i].Re, solns[i].Im);
    }

    fclose(fp);
}

// find the solutions each val in h_zVals is closest to, then output the index of that solution,
// along with the corresponding initial point to a CSV
void outputVals(Complex *zVals, Complex *h_zVals, Complex *h_solns, Complex *h_zValsInitial,
                int nSolns, std::string filename, int norm, int step=-1)
{
    dim3 B(16, 16, 1);
    dim3 G((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);

    int *closest;
    cudaMalloc(&closest, NRe*NIm*sizeof(int));

    Complex *solns;
    cudaMalloc(&solns, nSolns*sizeof(Complex));
    cudaMemcpy(solns, h_solns, nSolns*sizeof(Complex), cudaMemcpyHostToDevice);

    findClosestSoln <<< G, B >>> (closest, zVals, NRe, NIm, solns, nSolns, norm);

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

    FILE *fp = fopen(outputFilename.c_str(), "w");

    for (int i = 0; i < NRe*NIm; ++i)
        fprintf(fp, "%f, %f, %d\n", h_zValsInitial[i].Re, h_zValsInitial[i].Im, h_closest[i]);

    fclose(fp);

    cudaFree(closest); free(h_closest);
    cudaFree(solns);
}

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

    char *test = argv[3];

    Polynomial P;

    Complex *zValsInitial;
    Complex *zVals;
    int order;

    dfloat ReSpacing;
    dfloat ImSpacing;

    // test on -4x^3 + 6x^2 + 2x = 0, which has roots
    // 0, ~1.78, ~-.28
    if (strcmp(test, "smallTest") == 0 || strcmp(test, "smallTestL1") == 0)
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

    // random polynomial of order 7
    else if (strcmp(test, "bigTest") == 0 || strcmp(test, "bigTestL1") == 0)
    {
        int max = 10;
        int seed = 123456;
        order = 7;

        // create a random order 7 polynomial
        P = randomPolynomial(order, max, seed);

        ReSpacing = 4;
        ImSpacing = 4;
    }

    // order 12
    else if (strcmp(test, "bigTest2") == 0 || strcmp(test, "bigTest2L1") == 0)
    {
        // create a random order 11 polynomial
        int max = 50;
        int seed = 654321;

        order = 12;

        ReSpacing = 15;
        ImSpacing = 5;
        P = randomPolynomial(order, max, seed);
    }

    else if (strcmp(test, "bigTest3") == 0 || strcmp(test, "bigTest3L1") == 0)
    {
        // create a random order 11 polynomial
        int max = 100;
        int seed = 123456;

        order = 50;

        ReSpacing = 50;
        ImSpacing = 25;
        P = randomPolynomial(order, max, seed);
    }

    else
        return 0;

    // P' - derivative of P
    Polynomial Pprime = derivative(P);

    // device versions for newtonIterate
    Polynomial c_P      = deviceP(P);
    Polynomial c_Pprime = deviceP(Pprime);

    Complex *h_solns = (Complex *)malloc(order*sizeof(Complex));

    NRe = atoi(argv[1]);
    NIm = atoi(argv[2]);

    int N = NRe*NIm;

    dim3 B(16, 16, 1);
    dim3 G((NRe + 16 - 1)/16, (NRe + 16 - 1)/16);

    // arrays for initial points and points following iteration
    cudaMalloc(&zValsInitial, N*sizeof(Complex));
    cudaMalloc(&zVals,        N*sizeof(Complex));

    Complex *h_zValsInitial = (Complex *)malloc(N*sizeof(Complex));
    Complex *h_zVals        = (Complex *)malloc(N*sizeof(Complex));

    fillArrays <<< G, B >>> (ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);

    cudaMemcpy(h_zValsInitial, zValsInitial, N*sizeof(Complex), cudaMemcpyDeviceToHost);

    // perform 500 steps of the iteration and copy result back to host
    iterate(c_P, c_Pprime, 500, zVals, h_zVals);
    cudaMemcpy(h_zVals, zVals, N*sizeof(Complex), cudaMemcpyDeviceToHost);

    // find the solutions and output to CSV
    outputSolns(P, h_zVals, &h_solns, order, N, test);

    int norm;

    // output solutions to file and store them
    if (argc > 4 && strcmp(argv[4], "L1") == 0 ||
        argc > 5 && strcmp(argv[5], "L1") == 0)
        norm = 1;

    else
        norm = 2;

    if (argc > 4 && strcmp(argv[4], "step") == 0) {

        for (int i = 0; i < 100; ++i)
        {
            // output then perform 1 iteration
            outputVals(zVals, h_zVals, h_solns, h_zValsInitial, order, test, norm, i);
            iterate(c_P, c_Pprime, 1, zVals, h_zVals);
        }
    }

    else
    {
        // otherwise just do one output
        outputVals(zVals, h_zVals, h_solns, h_zValsInitial, order, test, norm);
    }


    cudaFree(zVals)          ; free(h_zVals)       ;
    cudaFree(zValsInitial)   ; free(h_zValsInitial);
    cudaFree(c_P.coeffs)     ; free(P.coeffs)      ;
    cudaFree(c_Pprime.coeffs);

    free(h_solns);
    return 0;
}
