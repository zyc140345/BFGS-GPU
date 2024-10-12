//
// Created by 张易诚 on 24-10-10.
//

#include "kernel.h"

__global__ void _FillVec_kernel(double *v, double value, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        v[tid] = value;
}

__global__ void _FillDiagonal_kernel(double *H, double value, int n) {
    // gridDim: (n_row, blocks_per_row)
    int vid = threadIdx.x + blockIdx.y * blockDim.x;
    while (vid < n) {
        if (vid == blockIdx.x)
            H[blockIdx.x * n + vid] = value;
        vid += blockDim.x * gridDim.y;
    }
}

__global__ void _CalcHy_kernel(const double *H, const double *y, double *Hy, int n, bool flip) {
    // gridDim: (n_row, blocks_per_row)
    __shared__ double cache[threadsPerBlock];
    int vid = threadIdx.x + blockIdx.y * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0;
    while (vid < n) {
        temp += H[blockIdx.x * n + vid] * y[vid];
        vid += blockDim.x * gridDim.y;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        if (flip)
            atomicAdd(Hy + blockIdx.x, -cache[0]);
        else
            atomicAdd(Hy + blockIdx.x, cache[0]);
    }
}

__global__ void _CalcyTH_kernel(const double *y, const double *H, double *yTH, int n) {
    // gridDim: (n_col, blocks_per_col)
    __shared__ double cache[threadsPerBlock];
    int vid = threadIdx.x + blockIdx.y * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0;
    while (vid < n) {
        temp += H[vid * n + blockIdx.x] * y[vid];
        vid += blockDim.x * gridDim.y;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        atomicAdd(yTH + blockIdx.x, cache[0]);
}

__global__ void _VecDot_kernel(const double *a, const double *b, double *c, int n) {
    __shared__ double cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0.0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        atomicAdd(c, cache[0]);
}

__global__ void _VecMult_kernel(double *v, double t, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        v[tid] *= t;
}

__global__ void _UpdateH_kernel(double *H, const double *s, const double *Hy,
                                const double *yTH, double sy, double tmp, int n) {
    // gridDim: (n_row, blocks_per_row)
    int i = blockIdx.x;
    int j = threadIdx.x + blockIdx.y * blockDim.x;

    double s_i = s[i];
    double Hy_i = Hy[i];

    while (j < n) {
        double s_j = s[j];
        double yTH_j = yTH[j];
        H[i * n + j] += (tmp * s_i * s_j - Hy_i * s_j - s_i * yTH_j) / sy;
        j += blockDim.x * gridDim.y;
    }
}