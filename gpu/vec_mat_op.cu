//
// Created by 张易诚 on 24-10-12.
//

#include "vec_mat_op.h"
#include "kernel.h"
#include "util.h"

void VecCopy(std::vector<double> &dst, const std::vector<double> &src) {
    auto n = src.size();
    for (int i = 0; i < n; i++)
        dst[i] = src[i];
}

void VecSub(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &ret) {
    auto n = a.size();
    for (int i = 0; i < n; i++)
        ret[i] = a[i] - b[i];
}

double VecDot(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0;
    auto n = a.size();
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

double VecDot(const double *dev_a, const double *dev_b, int n) {
    double result = 0.0;
    double *dev_result;
    CUDA_CHECK(cudaMalloc((void **) &dev_result, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dev_result, &result, sizeof(double), cudaMemcpyHostToDevice));

    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    _VecDot_kernel<<<num_blocks, threadsPerBlock>>>(dev_a, dev_b, dev_result, n);
    CUDA_CHECK(cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost));

    return result;
}

void VecMult(std::vector<double> &v, double t) {
    auto n = v.size();
    for (int i = 0; i < n; i++)
        v[i] *= t;
}

void VecMult(double *dev_v, double t, int n) {
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    _VecMult_kernel<<<num_blocks, threadsPerBlock>>>(dev_v, t, n);
}

void VecAxPy(const std::vector<double> &a, double t, const std::vector<double> &b, std::vector<double> &ret) {
    auto n = a.size();
    for (int i = 0; i < n; i++)
        ret[i] = a[i] + b[i] * t;
}

double VecLen(const std::vector<double> &v) {
    return sqrt(VecDot(v, v));
}

double VecLen(const double *dev_v, int n) {
    return sqrt(VecDot(dev_v, dev_v, n));
}

void VecNorm(std::vector<double> &v) {
    double tmp = VecLen(v);
    if (tmp > 0.0) {
        VecMult(v, 1.0 / tmp);
    }
}

void VecNorm(double *dev_v, int n) {
    double tmp = VecLen(dev_v, n);
    if (tmp > 0.0) {
        VecMult(dev_v, 1.0 / tmp, n);
    }
}

void FillVec(double *dev_v, double value, int n) {
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    _FillVec_kernel<<<num_blocks, threadsPerBlock>>>(dev_v, value, n);
}

void FillDiagonal(double *dev_H, double value, int n) {
    int blocksPerRow = imin(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blocks(n, blocksPerRow);
    _FillDiagonal_kernel<<<blocks, threadsPerBlock>>>(dev_H, value, n);
}

void UpdateH(double *dev_H, const double *dev_s, const double *dev_Hy,
             const double *dev_yTH, double sy, double tmp, int n) {
    int blocksPerRow = imin(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blocks(n, blocksPerRow);
    _UpdateH_kernel<<<blocks, threadsPerBlock>>>(
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