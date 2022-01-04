<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [CUDA Newton Fractals](#cuda-newton-fractals)
    - [Running the Code](#running-the-code)
    - [Creating mp4s of the Evolution](#creating-mp4s-of-the-evolution)
    - [Producing a Custom Fractal](#producing-a-custom-fractal)
    - [Dependencies](#dependencies)

<!-- markdown-toc end -->
# CUDA Newton Fractals

Visualizing the convergence of Newton's iteration using CUDA to asynchronously perform the iteration

***Note***: This project was my final project for CMDA 4984: SS Scientific Computing at Scale. I've been working on the project since then, but what I submitted for that project is in the `old` branch.

Newton's method is an iteration that (usually) converges to the roots of a polynomial. In this project, the iteration is performed on initial guesses evenly spaced all over the complex (Real and Imaginary) plane. Then, the roots those initial guesses converge to are color-coded based on which root they converged to. For example, initial guesses that converge to the first root we find may be yellow, guesses that converge to be closest to second root we find may be red, and so on. These are known as [Newton's Fractals](https://en.wikipedia.org/wiki/Newton_fractal).

For example:

[fractals/order12](fractals/order12.png):
![order12](fractals/order12.png)

A CUDA kernel in [src/newton.cu](src/newton.cu) performs the iteration asynchronously for each initial guess.

## Running the Code

Example use:

```bash
# compile
make

# run code - this will output testName.png in fractals/
./bin/newton <testName> [xPixels=?] [yPixels=?] [xRange=?] [yRange=?] [L1={true,false}] [step={true,false}]

## Or run my provided examples:
make runOrder7
make runOrder12
```

`testName` can be one of:

| Name          | Description                                       |
|--             |--                                                 |
| order7        | a given order 7 polynomial                        |
| order12       | a given order 12 polynomial - fractal shown above |
| anything else | you will be prompted to specify a polynomial      |

*Note*: If you use order7 or order12, xRange and yRange will already be set

| Parameter | Description                                                | Default |
|--         | --                                                         |--       |
| xPixels   | Number of horizontal pixels                                | 1000    |
| yPixels   | Number of vertical pixels                                  | 1000    |
| xyPixels  | Set both xPixels and yPixels                               | 1000    |
| xRange    | if 4, x values are between -4 and 4                        | 1.5     |
| yRange    | same as above but for the y values                         | 1.5     |
| xyRange   | set both xRange and yRange                                 | 1.5     |
| L1        | set to true if you want to use L1 norm to measure distance | false   |
| step      | set to true to output a png for each step                  | false   |

## Creating mp4s of the Evolution

You can also creat mp4s of the evolution of the fractals using:

```bash
make movie name=testName args="xyPixels=2000 xyRange=2"

## Or

# output pngs for each step
./bin/newton order12 step=true

# then stitch into a movie
make stitchMovie name=order12
```

This will output [order12.mp4](fractals/order12.mp4) in [fractals](fractals).

## Producing a Custom Fractal

You can also input your own polynomial, which will occur if `testName` isn't order7 or order12. You will be prompted to enter the roots of your polynomial, or 'random'. Example use:

```bash
$ ./bin/newton randomOrder50 xyPixels=2000 # higher resolution than defaults
Enter up to 99 characters of the roots of your polynomial separated by spaces:
ex. 5 4 3 2 1 to correspond to 5x^4 + 4x^3 + 3x^2 + 2x + 1
Or, enter 'random' to get a random polynomial
random
Enter [order] [max] [seed]
Order - the order of the polynomial
Max   - the max value of the coefficients (if 10, then all coefficients will be from -10 to 10
Seed  - seed the random polynomial (seeds drand48)
20 30 # an order 20 polynomial with roots between -30 and 30
```

This produces the fractal [fractals/randomOrder20.png](fractals/randomOrder20.png):

![randomOrder20.png](fractals/randomOrder20.png)

## Dependencies

| Dependency                  | Command to install on Ubuntu             |
|--                           |--                                        |
| Libpng                      | sudo apt-get install libpng-dev          |
| CUDA and a cuda-capable GPU | sudo apt-get install nvidia-cuda-toolkit |
| ffmpeg (makes the movies)   | sudo apt-get install ffmpeg              |
