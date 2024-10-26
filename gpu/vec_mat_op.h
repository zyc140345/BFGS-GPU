//
// Created by 张易诚 on 24-10-12.
//

#ifndef BFGS_GPU_VEC_MAT_OP_H
#define BFGS_GPU_VEC_MAT_OP_H

#include <vector>
#include <cublas_v2.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>

using mr = thrust::system::cuda::universal_host_pinned_memory_resource;
using pinned_allocator = thrust::mr::stateless_resource_allocator<double, mr>;
using pinned_vector = std::vector<double, pinned_allocator>;

void VecCopy(pinned_vector &dst, const pinned_vector &src);
void VecSub(const pinned_vector &a, const pinned_vector &b, pinned_vector &ret);
double VecDot(const pinned_vector &a, const pinned_vector &b);
double VecDot(cublasHandle_t h, const double *dev_a, const double *dev_b, int n);
void VecMult(pinned_vector &v, double t);
void VecAxPy(const pinned_vector &a, double t, const pinned_vector &b, pinned_vector &ret);
double VecLen(const pinned_vector &v);
void VecNorm(pinned_vector &v);
void VecNorm(cublasHandle_t h, double *dev_v, int n);
void FillVec(cudaStream_t s, double *dev_v, double value, int n);
void FillDiagonal(cudaStream_t s, double *dev_H, double value, int n);
void UpdateH(cudaStream_t s, double *dev_H, const double *dev_s, const double *dev_Hy,
             const double *dev_yTH, double sy, double tmp, int n);
void CalcyTH(cublasHandle_t h, const double *dev_y, const double *dev_H, double *dev_yTH, int n);
void CalcHy(cublasHandle_t h, const double *dev_H, const double *dev_y, double *dev_Hy, int n);
void Calcp(cublasHandle_t h, const double *dev_H, const double *dev_g, double *dev_p, int n);

#endif //BFGS_GPU_VEC_MAT_OP_H
