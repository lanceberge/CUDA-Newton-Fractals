#include "polynomial.h"
#include "complex.h"

// pre: coeffs is dynamically allocated with new
// post: c_coeffs is allocated with cudaMalloc
Polynomial::Polynomial(int order, dfloat *coeffs)
{
    this->order = order;
    h_coeffs = coeffs;

    cudaMalloc(&c_coeffs, (order+1)*sizeof(dfloat));

    // copy host coefficients to device array
    cudaMemcpy(c_coeffs, h_coeffs, (order+1)*sizeof(dfloat), cudaMemcpyHostToDevice);
}

// find the first derivative of a polynomial
Polynomial Polynomial::derivative()
{
    dfloat *derivative_coeffs = new dfloat[order];

    // update Pprime coeffs
    for (int i = 0; i < order; ++i) {
        derivative_coeffs[i] = h_coeffs[i]*(order-i);
    }

    return Polynomial(order - 1, derivative_coeffs);
}

// find P(z) - plug in a point z to the polynomial - on host coeffs
__host__ Complex Polynomial::h_Pz(const Complex& z) const
{
    dfloat ReSum = h_coeffs[order];
    dfloat ImSum = 0;

    // zPow on first iteration, then zPow^2, then ^3, etc.
    Complex zPow(z.Re, z.Im);

    // for A, B, C, D in coeffs. of P, return the cumulative sum of Az^4 + Bz^3 + ...
    for (int i = order-1; i >= 0; --i) {
        int coeff = h_coeffs[i];

        // zPow = z, then z^2, then z^3, etc.
        ReSum += coeff*zPow.Re;
        ImSum += coeff*zPow.Im;

        // update zPow to zPow*zPow
        zPow = zPow*z;
    }

    return Complex(ReSum, ImSum);
}

// find P(z) - plug in a point z to the polynomial - on device coeffs
__device__ Complex Polynomial::c_Pz(const Complex& z) const
{
    dfloat ReSum = c_coeffs[order];
    dfloat ImSum = 0;

    // zPow on first iteration, then zPow^2, then ^3, etc.
    Complex zPow(z.Re, z.Im);

    // for A, B, C, D in coeffs. of P, return the cumulative sum of Az^4 + Bz^3 + ...
    for (int i = order-1; i >= 0; --i) {
        int coeff = c_coeffs[i];

        // zPow = z, then z^2, then z^3, etc.
        ReSum += coeff*zPow.Re;
        ImSum += coeff*zPow.Im;

        // update zPow to zPow*zPow
        zPow = zPow*z;
    }

    return Complex(ReSum, ImSum);
}

// free memory
__host__ __device__ Polynomial::~Polynomial()
{
    #if !defined(__CUDACC__)
    delete[] h_coeffs;
    cudaFree(c_coeffs);
    #endif
}

// the coefficients of a random polynomial - coefficients are
// random between -max and max. seed is the seed for drand
dfloat *randomCoeffs(int order, int max, int seed)
{
    srand48(seed);

    dfloat *coeffs = new dfloat[order + 1];

    for (int i = 0; i < order + 1; ++i) {
        coeffs[i] = -max + 2*max*(drand48());
    }

    return coeffs;
}
