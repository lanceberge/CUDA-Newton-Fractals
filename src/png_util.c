#include <stdlib.h>
#include "png_util.h"

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
    // index 0-blue, 1-orange, 2-purple, 3-light green, 4-dark red, 5-light blue, 6-yellow,
    // 7-dark blue, 8-light purple, 9-light pink, dark green, yellow
    float r[12] = {0   , 0.85, 0.49, 0.47, 0.64, 0.30, 0.93, 0.05, 0.69, 1   , 0  ,  1};
    float g[12] = {0.45, 0.33, 0.18, 0.67, 0.08, 0.75, 0.69, 0.30, 0.51, 0.65, 0.37, 0};
    float b[12] = {0.74, 0.10, 0.56, 0.19, 0.18, 0.93, 0.13, 0.50, 0.85, 0.8 , 0.17, 0.1};

    int x, y;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < height; ++x) {
            png_byte *ptr = &(row[x*3]);

            int val = buffer[y*width + x];

            // if val is > 11, use random colors
            if (val < 12) {
                // convert into RGB triplets by multiplying each by 255
                ptr[0] = (int)(r[val]*255);
                ptr[1] = (int)(g[val]*255);
                ptr[2] = (int)(b[val]*255);
            }

            else {
                srand(val);
                ptr[0] = rand() % 255;
                ptr[1] = rand() % 255;
                ptr[2] = rand() % 255;
            }
        }

        png_write_row(png_ptr, row);
    }

    png_write_end(png_ptr, NULL);

    fclose(fp);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    free(row);
}
