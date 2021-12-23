#include "newton.h"
#include <png.h>
#include <stdlib.h>
#include <string>

void setRGB(png_byte *ptr, int val)
{
    // arrays for red, green, and blue percentages
    float r[12] = {0    , 0.85 , 0.494, 0.466, 0.635, 0.301, 0.929, 1, .69 , 1   , 0  , 0};
    float g[12] = {0.447, 0.325, 0.184, 0.674, 0.078, 0.745, 0.694, 0, 0.61, 0.75, 0.6, 0.5};
    float b[12] = {0.741, 0.098, 0.556, 0.188, 0.184, 0.933, 0.125, 0, 0.85, 0.8 , 0.3, 0.5};

    // convert into RGB triplets by multiplying each by 256
    ptr[0] = (int)(r[val]*256);
    ptr[1] = (int)(g[val]*256);
    ptr[2] = (int)(b[val]*256);
}

void writeImage(const char *filename, int width, int height, int *buffer, const char *title)
{
    FILE *fp = fopen(filename, "wb");

    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_bytep row = NULL;

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    info_ptr = png_create_info_struct(png_ptr);
    setjmp(png_jmpbuf(png_ptr));

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_BASE);

    // set title
    png_text title_text;
    title_text.compression = PNG_TEXT_COMPRESSION_NONE;
    title_text.key  = (char*)"Title";
    title_text.text = (char*)title;
    png_set_text(png_ptr, info_ptr, &title_text, 1);

    png_write_info(png_ptr, info_ptr);

    // write to png row by row
    row = (png_bytep)malloc(3 * width * sizeof(png_byte));

    int x, y;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < height; ++x) {
            setRGB(&(row[x * 3]), buffer[y * width + x]);
        }
        png_write_row(png_ptr, row);
    }

    png_write_end(png_ptr, NULL);

    fclose(fp);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    free(row);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
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
    if (strcmp(test, "smallTest") == 0 || strcmp(test, "smallTestL1") == 0) {
        order = 3;

        // create a polynomial
        dfloat *coeffs = new dfloat[4]{-4, 6, 2, 0};
        P.coeffs = coeffs;
        P.order = order;

        // the spacing on our grid, i.e. 1000 => run iteration on Nx and Ny evenly
        // spaced points from -1000 to 1000 on x and y
        ReSpacing = 4;
        ImSpacing = 4;
    }

    // random polynomial of order 7
    else if (strcmp(test, "bigTest") == 0 || strcmp(test, "bigTestL1") == 0) {
        int max = 10;
        int seed = 123456;
        order = 7;

        // create a random order 7 polynomial
        P = randomPolynomial(order, max, seed);

        ReSpacing = 4;
        ImSpacing = 4;
    }

    // order 12
    else if (strcmp(test, "bigTest2") == 0 || strcmp(test, "bigTest2L1") == 0) {
        // create a random order 11 polynomial
        int max = 50;
        int seed = 654321;

        order = 12;

        ReSpacing = 5;
        ImSpacing = 5;
        P = randomPolynomial(order, max, seed);
    }

    else
        return 0;

    // P' - derivative of P
    Polynomial Pprime = derivative(P);

    // device versions for newtonIterate
    Polynomial c_P = deviceP(P);
    Polynomial c_Pprime = deviceP(Pprime);

    Complex *h_solns = (Complex *)malloc(order * sizeof(Complex));

    int NRe = atoi(argv[1]);
    int NIm = atoi(argv[2]);
    int N = NRe * NIm;

    dim3 B(16, 16, 1);
    dim3 G((NRe + 16 - 1) / 16, (NRe + 16 - 1) / 16);

    // arrays for initial points and points following iteration
    cudaMalloc(&zValsInitial, N * sizeof(Complex));
    cudaMalloc(&zVals, N * sizeof(Complex));

    Complex *h_zValsInitial = (Complex *)malloc(N * sizeof(Complex));
    Complex *h_zVals = (Complex *)malloc(N * sizeof(Complex));

    fillArrays<<<G, B>>>(ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);

    cudaMemcpy(h_zValsInitial, zValsInitial, N * sizeof(Complex), cudaMemcpyDeviceToHost);

    // perform 500 steps of the iteration and copy result back to host
    newtonIterate<<<G, B>>>(zVals, c_P, c_Pprime, NRe, NIm, 500);

    // copy result to host
    cudaMemcpy(h_zVals, zVals, N * sizeof(Complex), cudaMemcpyDeviceToHost);

    // find the solutions - unique values in zVals
    h_solns = (Complex *)malloc(order * sizeof(Complex));
    int nSolns = findSolns(P, h_solns, h_zVals, order, N);

    int norm = (argc > 4 && strcmp(argv[4], "L1") == 0 ||
                argc > 5 && strcmp(argv[5], "L1") == 0) ? 1 : 2;

    // find closest solutions to each point in zVals
    int *closest;
    cudaMalloc(&closest, N * sizeof(int));

    Complex *solns;
    cudaMalloc(&solns, nSolns * sizeof(Complex));
    cudaMemcpy(solns, h_solns, nSolns * sizeof(Complex), cudaMemcpyHostToDevice);

    findClosestSoln <<<G, B>>> (closest, zVals, NRe, NIm, solns, nSolns, norm);

    // fill *closest with an integer corresponding to the solution its closest to
    // i.e. 0 for if this point is closest to solns[0]
    int *h_closest = (int *)malloc(N * sizeof(int));

    // copy results back to host
    cudaMemcpy(h_closest, closest, N * sizeof(int), cudaMemcpyDeviceToHost);

    writeImage(("plots/"+std::string(test)+".png").c_str(), NRe, NIm, h_closest, "closest");

    cudaFree(closest)        ; free(h_closest)     ;
    cudaFree(zVals)          ; free(h_zVals)       ;
    cudaFree(zValsInitial)   ; free(h_zValsInitial);
    cudaFree(c_P.coeffs)     ; free(P.coeffs)      ;
    cudaFree(c_Pprime.coeffs);
    cudaFree(solns)          ; free(h_solns)       ;
    return 0;
}
