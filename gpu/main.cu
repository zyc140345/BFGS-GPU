//
// Created by 张易诚 on 24-10-9.
//

#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <cublas_v2.h>

#include "vec_mat_op.h"
#include "util.h"

std::vector<int> objEqHeads, gradEqHeads;
std::vector<double> objEqVals, gradEqVals;

enum NodeType {
    NODE_CONST,
    NODE_OPER,
    NODE_VAR
};

enum OpType {
    OP_PLUS = 0,
    OP_MINUS = 1,
    OP_UMINUS = 2,
    OP_TIME = 3,
    OP_DIVIDE = 4,
    OP_SIN,
    OP_COS,
    OP_TG,
    OP_CTG,
    OP_SEC,
    OP_CSC,
    OP_ARCSIN,
    OP_ARCCOS,
    OP_ARCTG,
    OP_ARCCTG,
    OP_POW,
    OP_EXP,
    OP_EEXP,
    OP_SQR,
    OP_SQRT,
    OP_LOG,
    OP_LN,
    OP_NULL = -1
};

struct EqInfo {
    NodeType _type;
    double _val;
    int _var;
    OpType _op;
    int _left;
    int _right;
};

constexpr double epsZero1 = 1e-20;
constexpr double epsZero2 = 1e-7;
#ifndef M_PI_2
#define M_PI_2 (1.57079632679489661923)
#endif


#define BFGS_MAXIT 500
#define BFGS_STEP 0.1
#define BFGS_EPS 0.01
#define BFGS_MAXBOUND 1e+10


void CalcEqNew2(const pinned_vector &x, const std::vector<EqInfo> &etab,
                int st, int ed, std::vector<double> &vtab) {
    for (int i = ed - 1; i >= st; i--) {
        const EqInfo &eq = etab[i];
        switch (eq._type) {
            case NODE_CONST:
                vtab[i] = eq._val;
                break;

            case NODE_VAR: {
                int idx = eq._var;
                vtab[i] = x[idx];
                break;
            }

            case NODE_OPER: {
                double left = vtab[eq._left];
                double right = vtab[eq._right];
                switch (eq._op) {
                    case OP_PLUS:
                        vtab[i] = (left + right);
                        break;
                    case OP_MINUS:
                        vtab[i] = (left - right);
                        break;
                    case OP_UMINUS:
                        vtab[i] = -right;
                        break;
                    case OP_TIME:
                        vtab[i] = (left * right);
                        break;
                    case OP_DIVIDE:
                        vtab[i] = (left / right);
                        break;
                    case OP_SIN:
                        vtab[i] = (sin(left));
                        break;
                    case OP_COS:
                        vtab[i] = (cos(left));
                        break;
                    case OP_TG:
                        vtab[i] = (tan(left));
                        break;
                    case OP_CTG:
                        vtab[i] = (1.0 / tan(left));
                        break;
                    case OP_SEC:
                        vtab[i] = (1.0 / cos(left));
                        break;
                    case OP_CSC:
                        vtab[i] = (1.0 / sin(left));
                        break;
                    case OP_ARCSIN:
                        vtab[i] = (asin(left));
                        break;
                    case OP_ARCCOS:
                        vtab[i] = (acos(left));
                        break;
                    case OP_ARCTG:
                        vtab[i] = (atan(left));
                        break;
                    case OP_ARCCTG:
                        vtab[i] = (atan(-left) + M_PI_2);
                        break;
                    case OP_POW:
                        vtab[i] = (pow(left, right));
                        break;
                    case OP_EEXP:
                        vtab[i] = (exp(left));
                        break;
                    case OP_EXP:
                        vtab[i] = (exp(left * log(right)));
                        break;
                    case OP_LN:
                        vtab[i] = (log(left));
                        break;
                    case OP_LOG:
                        vtab[i] = (log(right) / log(left));
                        break;
                    case OP_SQR:
                        vtab[i] = (left * left);
                        break;
                    case OP_SQRT:
                        vtab[i] = (sqrt(left));
                        break;
                    default:
                        fprintf(stderr, "Unknown operator in EsCalcTree()\n");
                        assert(0);
                }
            }
        }
    }
}

double CalcEqNew1(const pinned_vector &x, const EqInfo &eq, const std::vector<EqInfo> &etab,
                  int item, const std::vector<int> &htab, int allNum, std::vector<double> &vtab) {
    int ed = item < 0 ? allNum : htab[item + 1];
    int st = item < 0 ? htab[-item] : htab[item];
    CalcEqNew2(x, etab, st, ed, vtab);

    switch (eq._type) {
        case NODE_OPER: {
            double left = vtab[eq._left];
            double right = vtab[eq._right];
            switch (eq._op) {
                case OP_PLUS:
                    return (left + right);
                case OP_MINUS:
                    return (left - right);
                case OP_UMINUS:
                    return (-right);
                case OP_TIME:
                    return (left * right);
                case OP_DIVIDE:
                    return (left / right);
                case OP_SIN:
                    return (sin(left));
                case OP_COS:
                    return (cos(left));
                case OP_TG:
                    return (tan(left));
                case OP_CTG:
                    return (1.0 / tan(left));
                case OP_SEC:
                    return (1.0 / cos(left));
                case OP_CSC:
                    return (1.0 / sin(left));
                case OP_ARCSIN:
                    return (asin(left));
                case OP_ARCCOS:
                    return (acos(left));
                case OP_ARCTG:
                    return (atan(left));
                case OP_ARCCTG:
                    return (atan(-left) + M_PI_2);
                case OP_POW:
                    return (pow(left, right));
                case OP_EEXP:
                    return (exp(left));
                case OP_EXP:
                    return (exp(left * log(right)));
                case OP_LN:
                    return (log(left));
                case OP_LOG:
                    return (log(right) / log(left));
                case OP_SQR:
                    return (left * left);
                case OP_SQRT:
                    return (sqrt(left));
                default:
                    fprintf(stderr, "Unknown operator in EsCalcTree()\n");
                    assert(0);
                    return (0.0);
            }
        }
    }

    assert(0);
    return 0;
}

double CalcObj(const pinned_vector &x, const std::vector<EqInfo> &eqs, int eqNum) {
    pinned_vector tmp;
    tmp.resize(eqNum);

#pragma omp parallel for num_threads(10)  // 无数据依赖，可并行
    for (int i = 0; i < eqNum; i++) {
        tmp[i] = CalcEqNew1(x, eqs[i], eqs, i == eqNum - 1 ? -i : i, objEqHeads, objEqVals.size(), objEqVals);
    }

    return VecDot(tmp, tmp);
}

void CalcGrad(const pinned_vector &x, pinned_vector &g, const std::vector<EqInfo> &eqs) {
    int n = x.size();

#pragma omp parallel for num_threads(10)  // 无数据依赖，可并行
    for (int i = 0; i < n; i++) {
        g[i] = CalcEqNew1(x, eqs[i], eqs, i == n - 1 ? -i : i, gradEqHeads, gradEqVals.size(), gradEqVals);
    }
}

double CalcObj(const pinned_vector &x0, double h, const pinned_vector &p,
               const std::vector<EqInfo> &eqs, int eqNum) {
    pinned_vector xt;
    xt.resize(x0.size());
    VecAxPy(x0, h, p, xt);
    return CalcObj(xt, eqs, eqNum);
}

void DetermineInterval(const pinned_vector &x0, double h, const pinned_vector &p,
                       double *left, double *right,
                       const std::vector<EqInfo> &eqs, int eqNum) {
    double A, B, C, D, u, v, w, s, r;

    A = CalcObj(x0, 0.0, p, eqs, eqNum);
    B = CalcObj(x0, h, p, eqs, eqNum);
    if (B > A) {
        s = -h;
        C = CalcObj(x0, s, p, eqs, eqNum);
        if (C > A) {
            *left = -h;
            *right = h;
            return;
        }
        B = C;
    } else {
        s = h;
    }
    u = 0.0;
    v = s;
    while (true) {
        s += s;
        if (fabs(s) > BFGS_MAXBOUND) {
            *left = *right = 0.0;
            return;
        }
        w = v + s;
        C = CalcObj(x0, w, p, eqs, eqNum);
        if (C >= B)
            break;
        u = v;
        A = B;
        v = w;
        B = C;
    }
    r = (v + w) * 0.5;
    D = CalcObj(x0, r, p, eqs, eqNum);
    if (s < 0.0) {
        if (D < B) {
            *left = w;
            *right = v;
        } else {
            *left = r;
            *right = u;
        }
    } else {
        if (D < B) {
            *left = v;
            *right = w;
        } else {
            *left = u;
            *right = r;
        }
    }
}

void GodenSep(const pinned_vector &x0, const pinned_vector &p,
              double left, double right, pinned_vector &x,
              const std::vector<EqInfo> &eqs, int eqNum) {
    static double beta = 0.61803398874989484820;
    double t1, t2, f1, f2;

    t2 = left + beta * (right - left);
    f2 = CalcObj(x0, t2, p, eqs, eqNum);
    ENTRY1:
    t1 = left + right - t2;
    f1 = CalcObj(x0, t1, p, eqs, eqNum);
    ENTRY2:
    if (fabs(t1 - t2) < epsZero2) {
        t1 = (t1 + t2) / 2.0;
        //printf("LineSearch t = %lf\n", t1*10000);

        VecAxPy(x0, t1, p, x);
        return;
    }
    if (fabs(left) > BFGS_MAXBOUND)
        return;
    if (f1 <= f2) {
        right = t2;
        t2 = t1;
        f2 = f1;
        goto ENTRY1;
    } else {
        left = t1;
        t1 = t2;
        f1 = f2;
        t2 = left + beta * (right - left);
        f2 = CalcObj(x0, t2, p, eqs, eqNum);
        goto ENTRY2;
    }
}

void LinearSearch(const pinned_vector &x0,
                  const pinned_vector &p,
                  double h,
                  pinned_vector &x,
                  const std::vector<EqInfo> &eqs,
                  int eqNum) {
    double left, right;

    DetermineInterval(x0, h, p, &left, &right, eqs, eqNum);
    if (left == right)
        return;

    //printf("%lf, %lf\n", left, right);
    GodenSep(x0, p, left, right, x, eqs, eqNum);
}

#define    H_EPS1    1e-5
#define    H_EPS2    1e-5
#define    H_EPS3    1e-4

bool HTerminate(const pinned_vector &xPrev,
                const pinned_vector &xNow,
                double fPrev, double fNow,
                const pinned_vector &gNow) {
    double ro;
    pinned_vector xDif(xNow.size());

    if (VecLen(gNow) >= H_EPS3)
        return false;

    VecSub(xNow, xPrev, xDif);
    ro = VecLen(xPrev);
    if (ro < H_EPS2)
        ro = 1.0;
    ro *= H_EPS1;
    if (VecLen(xDif) >= ro)
        return false;

    ro = fabs(fPrev);
    if (ro < H_EPS2)
        ro = 1.0;
    ro *= H_EPS1;
    fNow -= fPrev;
    if (fabs(fNow) >= ro)
        return false;

    return true;
}

void AnalysisEqs(const std::vector<EqInfo> &eqTab, int eqNum, std::vector<int> &eqHeads) {
    eqHeads.resize(eqNum);
    for (int i = 0; i < eqNum; i++) {
        const EqInfo &eq = eqTab[i];
        int left = eq._left;
        eqHeads[i] = left;
    }
}

int BFGSSolveEqs(char *data_path) {
    double eps = BFGS_EPS * BFGS_EPS;
    int itMax = BFGS_MAXIT;
    double step = BFGS_STEP;

    pinned_vector xNow, xKeep;
    std::vector<int> varMap, revMap;
    std::vector<EqInfo> objEqs;  // 目标函数
    int numObjEqs;
    std::vector<EqInfo> gradEqs;  // 目标函数导函数
    int numGradEqs;

    {
        FILE *fp = fopen(data_path, "rb");
        if (fp == NULL) {
            printf("%s failed to open for read.\n", data_path);
            return false;
        }

        int nx;
        fread(&nx, sizeof(int), 1, fp);
        xNow.resize(nx);  // 当前解
        fread(xNow.data(), sizeof(double), nx, fp);

        int n1, no;
        fread(&n1, sizeof(int), 1, fp);
        fread(&no, sizeof(int), 1, fp);
        numObjEqs = no;
        objEqs.resize(n1);  // 目标函数
        fread(objEqs.data(), sizeof(EqInfo), n1, fp);

        int ng;
        fread(&ng, sizeof(int), 1, fp);
        gradEqs.resize(ng);  // 目标函数导函数
        fread(gradEqs.data(), sizeof(EqInfo), ng, fp);
        numGradEqs = ng;

        int nk;
        fread(&nk, sizeof(int), 1, fp);
        assert(nk == nx);
        xKeep.resize(nk);
        fread(xKeep.data(), sizeof(double), nk, fp);

        //to remove recursive eval
        AnalysisEqs(objEqs, numObjEqs, objEqHeads);
        objEqVals.resize(objEqs.size());
        AnalysisEqs(gradEqs, nx, gradEqHeads);
        gradEqVals.resize(gradEqs.size());
    }

    // Do optimization
    double fNow = 0, fPrev = 0;
    int n = xNow.size();
    int itCounter = 0;

    pinned_vector gPrev, gNow, xPrev, p, y, s;  // p = H * g

    xPrev = xNow;
    gPrev.resize(n);
    gNow.resize(n);
    p.resize(n);
    y.resize(n);
    s.resize(n);

    double *dev_H, *dev_p, *dev_y, *dev_s, *dev_g, *dev_yTH, *dev_Hy;
    CUDA_CHECK(cudaMalloc((void **) &dev_H, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **) &dev_p, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **) &dev_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **) &dev_s, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **) &dev_g, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **) &dev_yTH, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **) &dev_Hy, n * sizeof(double)));

    cudaEvent_t g_start, g_stop;  // global
    CUDA_CHECK(cudaEventCreate(&g_start));
    CUDA_CHECK(cudaEventCreate(&g_stop));
    cudaEvent_t s_start, s_stop;  // step
    CUDA_CHECK(cudaEventCreate(&s_start));
    CUDA_CHECK(cudaEventCreate(&s_stop));

    cudaStream_t s1, s2;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));

    cublasHandle_t cublasH1, cublasH2;
    CUBLAS_CHECK(cublasCreate(&cublasH1));
    CUBLAS_CHECK(cublasCreate(&cublasH2));
    CUBLAS_CHECK(cublasSetStream(cublasH1, s1));
    CUBLAS_CHECK(cublasSetStream(cublasH2, s2));

    float t_total = 0.0;
    float t_step1 = 0.0;
    float t_step2 = 0.0;
    float t_step3 = 0.0;
    float t_step4 = 0.0;
    float t_step5 = 0.0;
    float t_step6 = 0.0;

//STEP1:
    RecordStartTime(g_start);
    RecordStartTime(s_start);
    fPrev = CalcObj(xNow, objEqs, numObjEqs);
    CalcGrad(xNow, gPrev, gradEqs);
    FillVec(s1, dev_H, 0.0, n * n);
    t_step1 += RecordStopTime(s_start, s_stop);

STEP2:
    RecordStartTime(s_start);
    FillDiagonal(s1, dev_H, 1.0, n);
    for (int i = 0; i < n; i++) {
        p[i] = -gPrev[i];
    }
    VecNorm(p);
    t_step2 += RecordStopTime(s_start, s_stop);

STEP3:
    RecordStartTime(s_start);
    if (itCounter++ > itMax) {
        t_step3 += RecordStopTime(s_start, s_stop);
        goto END;
    }

    xPrev = xNow;
    CUDA_CHECK(cudaStreamSynchronize(s1));
    LinearSearch(xPrev, p, step, xNow, objEqs, numObjEqs);
    fNow = CalcObj(xNow, objEqs, numObjEqs);
    std::cout << itCounter << " iterations, " << "f(x) = " << fNow << std::endl;

    if (fNow < eps) {
        t_step3 += RecordStopTime(s_start, s_stop);
        goto END;
    }

    CalcGrad(xNow, gNow, gradEqs);
    t_step3 += RecordStopTime(s_start, s_stop);

//STEP4:
    RecordStartTime(s_start);
    if (HTerminate(xPrev, xNow, fPrev, fNow, gNow)) {
        t_step4 += RecordStopTime(s_start, s_stop);
        goto END;
    }

//STEP5:
    RecordStartTime(s_start);
    if (fNow > fPrev) {
        VecCopy(xNow, xPrev);
        t_step5 += RecordStopTime(s_start, s_stop);
        goto STEP2;
    }

//STEP6:
    RecordStartTime(s_start);
    VecSub(gNow, gPrev, y);
    VecSub(xNow, xPrev, s);
    CUDA_CHECK(cudaMemcpyAsync(dev_y, y.data(), n * sizeof(double),
                               cudaMemcpyHostToDevice, s1));
    CUDA_CHECK(cudaMemcpyAsync(dev_s, s.data(), n * sizeof(double),
                               cudaMemcpyHostToDevice, s1));
    CUDA_CHECK(cudaMemcpyAsync(dev_g, gNow.data(), n * sizeof(double),
                               cudaMemcpyHostToDevice, s2));

    {
        double sy = VecDot(cublasH1, dev_s, dev_y, n);
        if (fabs(sy) < epsZero1) {
            t_step6 += RecordStopTime(s_start, s_stop);
            goto END;
        }

        CalcyTH(cublasH1, dev_y, dev_H, dev_yTH, n);
        CalcHy(cublasH2, dev_H, dev_y, dev_Hy, n);

        double tmp = VecDot(cublasH1, dev_yTH, dev_y, n);
        tmp = 1.0 + tmp / sy;
        CUDA_CHECK(cudaStreamSynchronize(s2));
        UpdateH(s1, dev_H, dev_s, dev_Hy, dev_yTH, sy, tmp, n);

        Calcp(cublasH1, dev_H, dev_g, dev_p, n);
        VecNorm(cublasH1, dev_p, n);
        CUDA_CHECK(cudaMemcpyAsync(p.data(), dev_p, n * sizeof(double),
                                   cudaMemcpyDeviceToHost, s1));

        fPrev = fNow;
        VecCopy(gPrev, gNow);
        VecCopy(xPrev, xNow);

        t_step6 += RecordStopTime(s_start, s_stop);
        goto STEP3;
    }

    END:
    std::cout << itCounter << " iterations" << std::endl;
    std::cout << "f(x) = " << fNow << std::endl;
    t_total += RecordStopTime(g_start, g_stop);

    printf("### Solver totally used %2.5f s ...\n", t_total);
    printf("    Step1 used %2.5f s\n", t_step1);
    printf("    Step2 used %2.5f s\n", t_step2);
    printf("    Step3 used %2.5f s\n", t_step3);
    printf("    Step4 used %2.5f s\n", t_step4);
    printf("    Step5 used %2.5f s\n", t_step5);
    printf("    Step6 used %2.5f s\n", t_step6);

    //Put results back...
    if (fNow < eps) {
        printf("Solved!!!!\n");
        return true;
    } else {
        printf("Solver Failed!!!!\n");
        return false;
    }

    CUDA_CHECK(cudaFree(dev_H));
    CUDA_CHECK(cudaFree(dev_p));
    CUDA_CHECK(cudaFree(dev_y));
    CUDA_CHECK(cudaFree(dev_s));
    CUDA_CHECK(cudaFree(dev_g));
    CUDA_CHECK(cudaFree(dev_yTH));
    CUDA_CHECK(cudaFree(dev_Hy));

    CUDA_CHECK(cudaEventDestroy(g_start));
    CUDA_CHECK(cudaEventDestroy(g_stop));
    CUDA_CHECK(cudaEventDestroy(s_start));
    CUDA_CHECK(cudaEventDestroy(s_stop));

    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
    CUBLAS_CHECK(cublasDestroy(cublasH1));
    CUBLAS_CHECK(cublasDestroy(cublasH2));
}

int main() {
    char data_path[] = "../../data/bfgs-large.dat";
    BFGSSolveEqs(data_path);
}

