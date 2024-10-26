//
// Created by 张易诚 on 24-10-12.
//

#include "vec_mat_op.h"
#include "kernel.h"
#include "util.h"

void VecCopy(pinned_vector &dst, const pinned_vector &src) {
    auto n = src.size();
    for (int i = 0; i < n; i++)
        dst[i] = src[i];
}

void VecSub(const pinned_vector &a, const pinned_vector &b, pinned_vector &ret) {
    auto n = a.size();
    for (int i = 0; i < n; i++)
        ret[i] = a[i] - b[i];
}

double VecDot(const pinned_vector &a, const pinned_vector &b) {
    double s = 0;
    auto n = a.size();
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

double VecDot(cublasHandle_t h, const double *dev_a, const double *dev_b, int n) {
    double result;
    CUBLAS_CHECK(cublasDdot(h, n, dev_a, 1, dev_b, 1, &result));
    return result;
}

void VecMult(pinned_vector &v, double t) {
    auto n = v.size();
    for (int i = 0; i < n; i++)
        v[i] *= t;
}

void VecAxPy(const pinned_vector &a, double t, const pinned_vector &b, pinned_vector &ret) {
    auto n = a.size();
    for (int i = 0; i < n; i++)
        ret[i] = a[i] + b[i] * t;
}

double VecLen(const pinned_vector &v) {
    return sqrt(VecDot(v, v));
}

void VecNorm(pinned_vector &v) {
    double tmp = VecLen(v);
    if (tmp > 0.0) {
        VecMult(v, 1.0 / tmp);
    }
}

void VecNorm(cublasHandle_t h, double *dev_v, int n) {
    double norm;
    CUBLAS_CHECK(cublasDnrm2(h, n, dev_v, 1, &norm));
    if (norm > 0.0) {
        double scale = 1.0 / norm;
        CUBLAS_CHECK(cublasDscal(h, n, &scale, dev_v, 1));
    }
}

void FillVec(cudaStream_t s, double *dev_v, double value, int n) {
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    _FillVec_kernel<<<num_blocks, threadsPerBlock, 0, s>>>(dev_v, value, n);
}

void FillDiagonal(cudaStream_t s, double *dev_H, double value, int n) {
    int blocksPerRow = imin(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blocks(n, blocksPerRow);
    _FillDiagonal_kernel<<<blocks, threadsPerBlock, 0, s>>>(dev_H, value, n);
}

void UpdateH(cudaStream_t s, double *dev_H, const double *dev_s, const double *dev_Hy,
             const double *dev_yTH, double sy, double tmp, int n) {
    int blocksPerRow = imin(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blocks(n, blocksPerRow);
    _UpdateH_kernel<<<blocks, threadsPerBlock, 0, s>>>(
            dev_H, dev_s, dev_Hy, dev_yTH, sy, tmp, n);
}

void CalcyTH(cublasHandle_t h, const double *dev_y, const double *dev_H, double *dev_yTH, int n) {
    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemv(h, CUBLAS_OP_N, n, n, &alpha,
                             dev_H, n, dev_y, 1, &beta, dev_yTH, 1));
}

void CalcHy(cublasHandle_t h, const double *dev_H, const double *dev_y, double *dev_Hy, int n) {
    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemv(h, CUBLAS_OP_T, n, n, &alpha,
                             dev_H, n, dev_y, 1, &beta, dev_Hy, 1));
}

void Calcp(cublasHandle_t h, const double *dev_H, const double *dev_g, double *dev_p, int n) {
    double alpha = -1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemv(h, CUBLAS_OP_T, n, n, &alpha,
                             dev_H, n, dev_g, 1, &beta, dev_p, 1));
}