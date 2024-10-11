//
// Created by 张易诚 on 24-10-10.
//

#ifndef BFGS_GPU_KERNEL_H
#define BFGS_GPU_KERNEL_H

constexpr int threadsPerBlock = 512;

__global__ void _FillVec_kernel(double *v, double value, int n);
__global__ void _Identity_kernel(double *H, int n);
__global__ void _CalcHy_kernel(const double *H, const double *y, double *Hy, int n, bool flip = false);
__global__ void _CalcyTH_kernel(const double *y, const double *H, double *yTH, int n);
__global__ void _VecDot_kernel(const double *a, const double *b, double *c, int n);
__global__ void _VecMult_kernel(double *v, double t, int n);
__global__ void _UpdateH_kernel(double *H, const double *s, const double *Hy,
                                const double *yTH, double sy, double tmp, int n);

#endif //BFGS_GPU_KERNEL_H
