//
// Created by 张易诚 on 24-10-9.
//

#ifndef BFGS_GPU_UTIL_H
#define BFGS_GPU_UTIL_H

#include <cstdio>
#include <cublas_v2.h>

void RecordStartTime(cudaEvent_t start);
float RecordStopTime(cudaEvent_t start, cudaEvent_t stop);

void CudaCheck(cudaError_t err, const char *file, int line);
void CublasCheck(cublasStatus_t err, const char *file, int line);
#define CUDA_CHECK(err) (CudaCheck(err, __FILE__, __LINE__))
#define CUBLAS_CHECK(err) (CublasCheck(err, __FILE__, __LINE__))

#define imin(a,b) (a<b?a:b)

#endif //BFGS_GPU_UTIL_H
