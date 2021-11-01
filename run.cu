#include "newton.h"

int main(int argc, int **argv)
{
    if (argc < 4)
    {
        printf("Usage: ./newton {Nx} {Ny} {Nit}\n");
        printf("Nx - Number of x-points to run iteration on\n");
        printf("Ny - Number of y-points to run iteration on\n");
        printf("Nit - Number of iterations to run\n");
        exit(-1);
    }

    int Nx = atoi(argv[2]);
    int Ny = atoi(argv[3]);
    int Nit = atoi(argv[4]);

    // the spacing on our grid, i.e. 1000 => run iteration on Nx and Ny evenly
    // spaced points from -1000 to 1000 on x and y
    int xSpacing = 1000;
    int ySpacing = 500;

    // total number of points
    int N = Nx*Ny;

    // an array of points
    PointChange **points = (PointChange*)malloc(Nx*Ny*sizeof(PointChange*));

    // starting x and y value
    int startX = 0 - xSpacing;
    int startY = 0 - ySpacing;

    // change in x and y at each iteration
    int dx = xSpacing*2 / Nx;
    int dy = ySpacing*2 / Ny;

    // TODO do this on device
    // TODO change points array to PointChange array

    // store evenly spaced Nx and Ny values in the range
    // -xySpacing to xySpacing in points
    for (int i = 0; i < Nx; ++i)
    {
        dfloat x = startX + i*dx;

        for (int j = 0; j < Ny; ++j)
        {
            dfloat y = startY + j*dy;

            Point *p = (Point*)malloc(sizeof(Point*));
            p->x = x;
            p->y = y;

            // store points in row-major format in points
            points[i + j*N] = p;
        }
    }

    return 0;
}
