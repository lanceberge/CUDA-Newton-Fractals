# CUDA Newton Fractals

Visualizing the convergence of Newton's iteration using CUDA to asynchronously perform the iteration

***Note***: This project was my final project for CMDA 4984: SS Scientific Computing at Scale. I've been working on the project since then, but what I submitted for that project is in the `old` branch.

Newton's method is used to find the roots of polynomials using an iterative sequences that (usually) converges to those roots. In this project, the iteration is performed on initial guesses evenly spaced all over the complex (Real and Imaginary) plane. Then, the roots those initial guesses converge to are color-coded based on which root they converged to. For example, initial guesses that converge to the first root we find may be yellow, guesses that converge to the second root we find may be red, and so on. These are known as [Newton's Fractals](https://en.wikipedia.org/wiki/Newton_fractal).

For example:

[fractals/order12](fractals/order12.png):
![order12](fractals/order12.png)

A CUDA kernel in [src/newton.cu](src/newton.cu) performs the iteration asynchronously for each initial guess.

## Running the Code

Example use:

```bash
# compile
make

# run code
./bin/newton <testName> [xPixels=500] [yPixels=500] [xRange=3] [yRange=3] [L1=false] [step=false]

## Or run my provided examples:
make runOrder7
make runOrder12
```

*Note*: The values after = can be set by you, 500, 3, false, etc. are just the defaults

`testName` can be one of:

| Name          | Description                                       |
|--             |--                                                 |
| order7        | a given order 7 polynomial                        |
| order12       | a given order 12 polynomial - fractal shown above |
| anything else | you will be prompted to specify a polynomial      |

*Note*: If you use order7 or order12, xRange and yRange will already be set

| Parameter | Description                                                |
|--         | --                                                         |
| xPixels   | Number of horizontal pixels                                |
| yPixels   | Number of vertical pixels                                  |
| xRange    | if 4, x values are between -4 and 4                        |
| yRange    | same as above but for the y values                         |
| L1        | set to true if you want to use L1 norm to measure distance |
| step      | set to true to output a png for each step                  |

For example:

```bash
./bin/newton order7 L1=true step=false
```
This will output a png in [fractals](fractals)

## Creating mp4s of the Evolution

You can also creat mp4s of the evolution of the fractals using:

```bash
make movie name=testName
```

*Note*: This will keep all the default parameters (xPixels, xRange, etc.). If you want to change those, first run the executable with step=true, then stitch into a movie with `make stitchMovie`, for example:

```bash
# output pngs for each step
./bin/newton order12 step=true xPixels=1000

# stitch into a movie
make stitchMovie name=order12

## Or using default parameters
make movie name=order12
```

This will output [order7.mp4](fractals/order7.mp4) in [fractals](fractals).

## Producing a Custom Fractal

You can also input your own polynomial, which will occur if `testName` isn't order7 or order12. You will be prompted to enter the roots of your polynomial, or 'random'. Example use:

```bash
$ ./bin/newton random xRange=1.5 yRange=1.5 xPixels=1000 yPixels=1000
Enter up to 99 characters of the roots of your polynomial separated by spaces:
ex. 5 4 3 2 1 to correspond to 5x^4 + 4x^3 + 3x^2 + 2x + 1
Or, enter 'random' to get a random polynomial
random
Enter [order] [max] [seed]
Order - the order of the polynomial
Max   - the max value of the coefficients (if 10, then all coefficients will be from -10 to 10
Seed  - seed the random polynomial (seeds drand48)
10 20 # an order 10 polynomial with roots between -20 and 20
```

This produces the fractal [fractals/random.png](fractals/random.png):

![random.png](fractals/random.png)

*Note*: The default xRange and yRange is 3. I set them to 1.5 for this fractal, as the default was too zoomed out. In other words, decreasing them zoomed everything in.

## Dependencies

| Dependency                  | Command to install on Ubuntu             |
|--                           |--                                        |
| Libpng                      | sudo apt-get install libpng-dev          |
| CUDA and a cuda-capable GPU | sudo apt-get install nvidia-cuda-toolkit |
| ffmpeg (makes the movies)   | sudo apt-get install ffmpeg              |
