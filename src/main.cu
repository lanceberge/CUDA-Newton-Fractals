#include "newton.h"
#include <png.h>
#include <stdlib.h>
#include <string>

// Output the data to png
// Essentially taken from: http://www.labbookpages.co.uk/software/imgProc/files/libPNG/makePNG.c
void writeImage(const char *filename, int width, int height, int *buffer)
{
    FILE *fp = fopen(filename, "wb");

    // initialize some pointers
    png_structp png_ptr = NULL;
    png_infop info_ptr  = NULL;
    png_bytep row       = NULL;

    // set up png and info ptr
    png_ptr  = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);

    setjmp(png_jmpbuf(png_ptr));

    png_init_io(png_ptr, fp);

    // set some metadata
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    // write to png row by row
    row = (png_bytep)malloc(3*width*sizeof(png_byte));

    // arrays for red, green, and blue percentages
    float r[12] = {0    , 0.85 , 0.494, 0.466, 0.635, 0.301, 0.929, 1, 0.69, 1   , 0  , 0};
    float g[12] = {0.447, 0.325, 0.184, 0.674, 0.078, 0.745, 0.694, 0, 0.61, 0.75, 0.6, 0.5};
    float b[12] = {0.741, 0.098, 0.556, 0.188, 0.184, 0.933, 0.125, 0, 0.85, 0.8 , 0.3, 0.5};

    int x, y;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < height; ++x) {
            png_byte *ptr = &(row[x*3]);

            int val = buffer[y*width + x];

            // convert into RGB triplets by multiplying each by 256
            ptr[0] = (int)(r[val]*256);
            ptr[1] = (int)(g[val]*256);
            ptr[2] = (int)(b[val]*256);
        }

        png_write_row(png_ptr, row);
    }

    png_write_end(png_ptr, NULL);

    fclose(fp);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    free(row);
}

// perform the iteration and output to png
int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Example Usage: ./bin/newton <testName> [NRe=300] NIm=300]\n"
               "               [ReSpacing=3] [ImSpacing=3] [L1=false] [step=false] \n");

        printf("testName  - name of the test, if bigTest or bigTest2, the other "
                "options will be ignored\n");
        printf("NRe       - Number of real points to run iteration on\n");
        printf("NIm       - number of imaginary points to run iteration on\n");
        printf("ReSpacing - if 4, then the real values will be spaced from -4 to 4\n");
        printf("ImSpacing - same as ReSpacing but for the imaginary values\n");
        printf("L1        - true if you want to use L1 norm to measure distance\n");
        printf("step      - true if you want to output a png for each step\n");
        exit(-1);
    }

    // device array pointers
    int     *closest;
    Complex *solns;
    Complex *zValsInitial;
    Complex *zVals;

    // will be initialized below based on which test we use
    dfloat ReSpacing = 3;
    dfloat ImSpacing = 3;
    int norm         = 2;
    bool step        = false;
    int NRe = 500;
    int NIm = 500;

    // characteristics of the polynomial
    int order;
    dfloat *h_coeffs;

    char *testName = argv[1];

    // iterate through command line args and set values
    for (int i = 2; i < argc; ++i) {
        // the value to set - i.e. NRe, L1, step
        char *token = strtok(argv[i], "=");

        // what to set it to
        char *val = strtok(NULL, "=");

        if (token != NULL && val != NULL) {
            if (strcmp(token, "L1") == 0)
                norm = strcmp(val, "true") == 0 ? 1 : 2;

            else if (strcmp(token, "step") == 0)
                step = strcmp(val, "true") == 0 ? true : false;

            else if (strcmp(token, "NRe") == 0)
                // if nothing is specified, set to 3, else to the specified value
                NRe = atoi(val);

            else if (strcmp(token, "NIm") == 0)
                NIm = atoi(val);

            else if (strcmp(token, "ReSpacing") == 0)
                ReSpacing = atoi(val);

            else if (strcmp(token, "ImSpacing") == 0)
                ImSpacing = atoi(val);
        }
    }


    // based on testName - either set the default values for these polynomials, or
    // prompt for a custom polynomial

    // test on -4x^3 + 6x^2 + 2x = 0, which has roots
    // 0, ~1.78, ~-.28
    if (strcmp(testName, "smallTest") == 0) {

        // create an order 3 polynomial
        order = 3;
        h_coeffs = new dfloat[4] {-4, 6, 2, 0};

        // the spacing on our grid, i.e. 1000 => run iteration on Nx and Ny evenly
        // spaced points from -1000 to 1000 on x and y
        ReSpacing = 4;
        ImSpacing = 4;
    }

    // random polynomial of order 7
    else if (strcmp(testName, "bigTest") == 0) {
        int max = 10;
        int seed = 123456;
        order = 7;

        // create a random order 7 polynomial
        h_coeffs = randomCoeffs(order, max, seed);

        if (!step) {
            NRe = 1000;
            NIm = 1000;
        }
    }

    // order 12
    else if (strcmp(testName, "bigTest2") == 0) {
        // create a random order 11 polynomial
        int max = 50;
        int seed = 654321;

        order = 12;

        ReSpacing = 2;
        ImSpacing = 2;
        h_coeffs = randomCoeffs(order, max, seed);

        if (!step) {
            NRe = 1000;
            NIm = 1000;
        }
    }

    // prompt for custom polynomial
    else {
        char str[100];

        do {
            printf("Enter up to 99 characters of the roots of your polynomial separated by spaces:\n"
                    "ex. 5 4 3 2 1 to correspond to 5x^4 + 4x^3 + 3x^2 + 2x + 1\n");

            printf("Or, enter 'random' to get a random polynomial\n");
        } while (scanf(" %99[^\n]", str) != 1);

        char *val = strtok(str, " ");

        // if random was entered, prompt for an order, max, and seed
        if (strcmp(val, "random") == 0) {
            do {
                printf("Enter [order] [max] [seed]\n");
                printf("Order - the order of the polynomial\n");
                printf("Max   - the max value of the coefficients (if 10, then all "
                        "coefficients will be from -10 to 10\n");
                printf("Seed  - seed the random polynomial (seeds drand48)\n");

            } while (scanf(" %99[^\n]", str) != 1);

            order = 5;
            int max  = 10;
            int seed = 123456;

            char *val = strtok(str, " ");
            if (val != NULL)
                order = atoi(val);

            val = strtok(NULL, " ");
            if (val != NULL)
                order = atoi(val);

            val = strtok(NULL, " ");
            if (val != NULL)
                seed = atoi(val);

            h_coeffs = randomCoeffs(order, max, seed);
        }

        // parse each entry separated by spaces into coeffs
        else {
            h_coeffs = new dfloat[12];

            int i;
            for (i = 0; i < 12 && val != NULL; ++i) {
                h_coeffs[i] = atof(val);
                val = strtok(NULL, " ");
            }

            order = i - 1;
        }
    }

    // create our polynomial and its first derivative
    Polynomial P(order, h_coeffs);

    Polynomial Pprime = P.derivative();
    int N = NRe * NIm;

    dim3 B(16, 16, 1);
    dim3 G((NRe + 16 - 1) / 16, (NRe + 16 - 1) / 16);

    // arrays for initial points and points following iteration
    cudaMalloc(&zValsInitial, N * sizeof(Complex));
    cudaMalloc(&zVals       , N * sizeof(Complex));

    Complex *h_zVals = (Complex *)malloc(N*sizeof(Complex));

    // initialize arrays - evenly spaced over complex plane
    fillArrays<<<G, B>>>(ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);

    // perform 500 steps of the iteration and copy result back to host
    newtonIterate<<<G, B>>>(zVals, P, Pprime, NRe, NIm, 500);

    // copy result to host
    cudaMemcpy(h_zVals, zVals, N * sizeof(Complex), cudaMemcpyDeviceToHost);

    // find the solutions - unique values in zVals
    Complex *h_solns = (Complex *)malloc(order*sizeof(Complex));
    int nSolns = findSolns(P, h_solns, h_zVals, order, N);

    free(h_zVals);

    // find closest solutions to each point in zVals
    cudaMalloc(&closest, N * sizeof(int));

    // copy h_solns to device for use in findClosestSoln
    cudaMalloc(&solns, nSolns * sizeof(Complex));
    cudaMemcpy(solns, h_solns, nSolns * sizeof(Complex), cudaMemcpyHostToDevice);

    free(h_solns);

    // array for closest solutions
    int *h_closest;
    h_closest = new int[N];

    // loop over 50 steps and output an image for each
    if (step) {
        // reset zVals
        fillArrays<<<G, B>>>(ReSpacing, ImSpacing, zValsInitial, zVals, NRe, NIm);

        for (int i = 0; i < 50; ++i) {
            // find the closest solution to each value in zVals and store it in closest
            findClosestSoln<<<G, B>>>(closest, zVals, NRe, NIm, solns, nSolns, norm);

            // copy results back to host
            cudaMemcpy(h_closest, closest, N * sizeof(int), cudaMemcpyDeviceToHost);

            // output image
            writeImage(("fractals/"+std::string(testName)+"Step-"+std::to_string(i)+".png").c_str(),
                       NRe, NIm, h_closest);

            // perform 1 iteration
            newtonIterate<<<G, B>>>(zVals, P, Pprime, NRe, NIm, 1);
        }
    }

    // just output the one image
    else {
        // find the closest solution to each value in zVals and store it in closest
        findClosestSoln<<<G, B>>>(closest, zVals, NRe, NIm, solns, nSolns, norm);

        // copy results back to host
        cudaMemcpy(h_closest, closest, N * sizeof(int), cudaMemcpyDeviceToHost);

        // output image
        writeImage(("fractals/"+std::string(testName)+".png").c_str(), NRe, NIm, h_closest);
    }

    // free heap memory
    delete[] h_closest;

    cudaFree(closest);
    cudaFree(solns);
    cudaFree(zVals);
    cudaFree(zValsInitial);

    return 0;
}
