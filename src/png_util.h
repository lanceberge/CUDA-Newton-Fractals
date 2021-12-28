#pragma once

#include <png.h>

// write data stored in buffer (ints between 1 and 12) to a png with width and height
void writeImage(const char *filename, int width, int height, int *buffer);
