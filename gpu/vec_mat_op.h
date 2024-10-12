//
// Created by 张易诚 on 24-10-12.
//

#ifndef BFGS_GPU_VEC_MAT_OP_H
#define BFGS_GPU_VEC_MAT_OP_H

#include <vector>
#include <cublas_v2.h>

void VecCopy(std::vector<double> &dst, const std::vector<double> &src);
void VecSub(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &ret);
double VecDot(const std::vector<double> &a, const std::vector<double> &b);
double VecDot(const double *dev_a, const double *dev_b, int n);
void VecMult(std::vector<double> &v, double t);
void VecMult(double *dev_v, double t, int n);
void VecAxPy(const std::vector<double> &a, double t, const std::vector<double> &b, std::vector<double> &ret);
double VecLen(const std::vector<double> &v);
double VecLen(const double *dev_v, int n);
void VecNorm(std::vector<double> &v);
void VecNorm(double *dev_v, int n);
void FillVec(double *dev_v, double value, int n);
void FillDiagonal(double *dev_H, double value, int n);
void UpdateH(double *dev_H, const double *dev_s, const double *dev_Hy,
             const double *dev_yTH, double sy, double tmp, int n);
void CalcyTH(cublasHandle_t h, const double *dev_y, const double *dev_H, double *dev_yTH, int n);
void CalcHy(cublasHandle_t h, const double *dev_H, const double *dev_y, double *dev_Hy, int n);
void Calcp(cublasHandle_t h, const double *dev_H, const double *dev_g, double *dev_p, int n);

#endif //BFGS_GPU_VEC_MAT_OP_H
