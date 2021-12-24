# CUDA Newton Iteration

Visualizing the convergence of Newton's iteration using CUDA to asynchronously perform the iteration

***Note***: This project was my final project for CMDA 4984: SS Scientific Computing at Scale. I've been working on the project since then, but what I submitted for that project is in the `old` branch.

The plots (located in [plots](plots)) are color coded based on which root of a polynomial the initial guess converged to be closest to

For example:

![bigTestPlot](plots/bigTest.png)

The iteration was performed on initial guesses evenly spaced over the complex plane. Each color corresponds to which root the initial guess there converged to be closest to. For example, yellow points mean that the initial guesses at those points converged to be closest to the yellow root (circled with a black outline). A CUDA kernel in [src/newton.cu](src/newton.cu) performs the iteration asynchronously for each initial guess.

## Running the Kernel

Example use:

```bash
make               # compile

./bin/newton <testName> [NRe=300] NIm=300] [ReSpacing=3] [ImSpacing=3]
                        [L1=false] [step=false]
```

`testName` can be one of:

| Name          | Description                                  |
|--             |                                              |
| bigTest       | a given order 7 polynomial                   |
| bigTest2      | a given order 12 polynomial                  |
| anything else | you will be prompted to specify a polynomial |

All of the other parameters are optional

*Note*: If you use bigTest or bigTest2, NRe, NIm, ReSpacing, and ImSpacing will already be set

| Parameter | Description                                                |
|--         |                                                            |
| NRe       | Number of real initial guess to run iteration on           |
| NIm       | Number of imaginary guesses                                |
| ReSpacing | if 4, the real initial guesses will be spaced from -4 to 4 |
| ImSpacing | same as above but for the imaginary values                 |
| L1        | set to true if you want to use L1 norm to measure distance |
| step      | set to true to output a png for each step                  |

For example:

```bash
./bin/newton bigTest L1=true step=false
```

This will output a png in plots, or 50 pngs if you set step to true

## Stitching into a Movie

If you set step to true, to stitch all of the pngs into a movie, run:

```bash
make movie name=testName
```

i.e., if you ran:

```bash
./bin/newton bigTest step=true
``

Then stitch into a movie with

```bash
make movie name=bigTest
```

This will output bigTest.mp4 in `plots`

## Dependencies

| Libpng                      | sudo apt-get install libpng-dev          |
| CUDA and a cuda-capable GPU | sudo apt-get install nvidia-cuda-toolkit |
| ffmpeg (makes the movies)   | sudo apt-get install ffmpeg              |
