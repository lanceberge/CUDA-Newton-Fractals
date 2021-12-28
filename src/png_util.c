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

    // arrays for red   , green, and blue percentages
    // blue, orange, purple, light green, dark red, light blue, yellow, dark blue, ... TODO
    float r[12] = {0    , 0.85 , 0.494, 0.466, 0.635, 0.301, 0.929, 0.10   , 0.69, 1   , 0  , 0};
    float g[12] = {0.447, 0.325, 0.184, 0.674, 0.078, 0.745, 0.694, 0.30   , 0.61, 0.75, 0.6, 0.5};
    float b[12] = {0.741, 0.098, 0.556, 0.188, 0.184, 0.933, 0.125, 0.60, 0.85, 0.8 , 0.3, 0.5};

    int x, y;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < height; ++x) {
            png_byte *ptr = &(row[x*3]);

            int val = buffer[y*width + x];

            // convert into RGB triplets by multiplying each by 256
            ptr[0] = (int)(r[val]*255);
            ptr[1] = (int)(g[val]*255);
            ptr[2] = (int)(b[val]*255);
        }

        png_write_row(png_ptr, row);
    }

    png_write_end(png_ptr, NULL);

    fclose(fp);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    free(row);
}
