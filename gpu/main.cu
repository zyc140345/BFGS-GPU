//
// Created by 张易诚 on 24-10-9.
//

#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

#include "../array2.h"
#include "util.h"
#include "kernel.h"

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

typedef struct _EqInfo {
    NodeType _type;
    double _val;
    int _var;
    OpType _op;
    int _left;
    int _right;
} EqInfo;

enum VarType {
    VAR_CONST,
    VAR_UNSOLVED,
    VAR_SOLVED,
    VAR_DELETED,
    VAR_FREE
};

enum OptimType {
    BFGS,
    LBFGS
};

struct VarInfo {
    VarType	_type;
    double _val;

    VarInfo(VarType ty, double val) : _type(ty), _val(val) {}
};

double epsZero1 = 1e-20;
double epsZero2 = 1e-7;
#ifndef M_PI_2
#define M_PI_2 (1.57079632679489661923)
#endif


#define		BFGS_MAXIT	318
#define		BFGS_STEP	0.1


static int _GetMaxIt()
{
    return BFGS_MAXIT;
}

static double _GetStep()
{
    return BFGS_STEP;
}

static double _GetEps()
{
    return 0.01;
}

static void _ConstructVarTab(std::vector<double>& vars, std::vector<int>& varMap, std::vector<int>& revMap);
static void _ConstructObjEqTab(std::vector<EqInfo>& eqs, int& numEqs, const std::vector<int>& revMap);
static void _ConstructGradEqTab(std::vector<EqInfo>& eqs, int& numEqs, const std::vector<int>& revMap);
static void _ScatterVarTab(std::vector<double>& x, std::vector<int>& varMap);

static void _VecCopy(std::vector<double>& dst, const std::vector<double>& src)
{
    int n = src.size();
    for (int i = 0; i < n; i++)
        dst[i] = src[i];
}

static void _VecSub(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& ret)
{
    int n = a.size();
    for (int i = 0; i < n; i++)
        ret[i] = a[i] - b[i];
}

static double _VecDot(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0;
    int n = a.size();
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }

    return s;
}

static double _VecDot(const double *dev_a, const double *dev_b, int n)
{
    double result = 0.0;
    double *dev_result;
    HANDLE_ERROR(cudaMalloc((void **) &dev_result, sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(dev_result, &result, sizeof(double), cudaMemcpyHostToDevice));

    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    _VecDot_kernel<<<num_blocks, threadsPerBlock>>>(dev_a, dev_b, dev_result, n);
    HANDLE_ERROR(cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost));

    return result;
}

static void _VecMult(std::vector<double>& v, double t)
{
    int n = v.size();
    for (int i = 0; i < n; i++)
        v[i] *= t;
}

static void _VecMult(double *dev_v, double t, int n)
{
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    _VecMult_kernel<<<num_blocks, threadsPerBlock>>>(dev_v, t, n);
}

static double _VecAdd(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& ret)
{
    int n = a.size();
    for (int i = 0; i < n; i++)
        ret[i] = a[i] + b[i];
}

static void _VecAxPy(const std::vector<double>& a, double t, const std::vector<double>& b, std::vector<double>& ret)
{
    int n = a.size();
    for (int i = 0; i < n; i++)
        ret[i] = a[i] + b[i] * t;
}

static double _VecLen(const std::vector<double>& v)
{
    return sqrt(_VecDot(v, v));
}

static double _VecLen(const double *dev_v, int n)
{
    return sqrt(_VecDot(dev_v, dev_v, n));
}

static void _VecNorm(std::vector<double>& v)
{
    double tmp = _VecLen(v);
    if (tmp > 0.0) {
        _VecMult(v, 1.0 / tmp);
    }
}

static void _VecNorm(double *dev_v, int n)
{
    double tmp = _VecLen(dev_v, n);
    if (tmp > 0.0) {
        _VecMult(dev_v, 1.0 / tmp, n);
    }
}

static void
_CalcEqNew2(const std::vector<double>& x, const std::vector<EqInfo>& etab, int st, int ed, std::vector<double>& vtab)
{
    for (int i = ed - 1; i >= st; i--) {
        const EqInfo& eq = etab[i];
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
                    case	OP_PLUS:
                        vtab[i] = (left + right);
                        break;
                    case	OP_MINUS:
                        vtab[i] = (left - right);
                        break;
                    case	OP_UMINUS:
                        vtab[i] = -right;
                        break;
                    case	OP_TIME:
                        vtab[i] = (left * right);
                        break;
                    case	OP_DIVIDE:
                        vtab[i] = (left / right);
                        break;
                    case	OP_SIN:
                        vtab[i] = (sin(left));
                        break;
                    case	OP_COS:
                        vtab[i] = (cos(left));
                        break;
                    case	OP_TG:
                        vtab[i] = (tan(left));
                        break;
                    case	OP_CTG:
                        vtab[i] = (1.0 / tan(left));
                        break;
                    case	OP_SEC:
                        vtab[i] = (1.0 / cos(left));
                        break;
                    case	OP_CSC:
                        vtab[i] = (1.0 / sin(left));
                        break;
                    case	OP_ARCSIN:
                        vtab[i] = (asin(left));
                        break;
                    case	OP_ARCCOS:
                        vtab[i] = (acos(left));
                        break;
                    case	OP_ARCTG:
                        vtab[i] = (atan(left));
                        break;
                    case	OP_ARCCTG:
                        vtab[i] = (atan(-left) + M_PI_2);
                        break;
                    case	OP_POW:
                        vtab[i] = (pow(left, right));
                        break;
                    case	OP_EEXP:
                        vtab[i] = (exp(left));
                        break;
                    case	OP_EXP:
                        vtab[i] = (exp(left * log(right)));
                        break;
                    case	OP_LN:
                        vtab[i] = (log(left));
                        break;
                    case	OP_LOG:
                        vtab[i] = (log(right) / log(left));
                        break;
                    case	OP_SQR:
                        vtab[i] = (left * left);
                        break;
                    case	OP_SQRT:
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

static double _CalcEqNew1(const std::vector<double>& x, const EqInfo& eq, const std::vector<EqInfo>& etab,
                          int item, const std::vector<int> &htab, int allNum, std::vector<double> &vtab)
{
    int ed = item < 0 ? allNum : htab[item + 1];
    int st = item < 0 ? htab[-item] : htab[item];
    _CalcEqNew2(x, etab,  st, ed, vtab);

    switch (eq._type) {
        case NODE_OPER: {
            double left = vtab[eq._left];
            double right = vtab[eq._right];
            switch (eq._op) {
                case	OP_PLUS:
                    return(left + right);
                case	OP_MINUS:
                    return(left - right);
                case	OP_UMINUS:
                    return(-right);
                case	OP_TIME:
                    return(left * right);
                case	OP_DIVIDE:
                    return(left / right);
                case	OP_SIN:
                    return(sin(left));
                case	OP_COS:
                    return(cos(left));
                case	OP_TG:
                    return(tan(left));
                case	OP_CTG:
                    return(1.0 / tan(left));
                case	OP_SEC:
                    return(1.0 / cos(left));
                case	OP_CSC:
                    return(1.0 / sin(left));
                case	OP_ARCSIN:
                    return(asin(left));
                case	OP_ARCCOS:
                    return(acos(left));
                case	OP_ARCTG:
                    return(atan(left));
                case	OP_ARCCTG:
                    return(atan(-left) + M_PI_2);
                case	OP_POW:
                    return(pow(left, right));
                case	OP_EEXP:
                    return(exp(left));
                case	OP_EXP:
                    return(exp(left * log(right)));
                case	OP_LN:
                    return(log(left));
                case	OP_LOG:
                    return(log(right) / log(left));
                case	OP_SQR:
                    return(left * left);
                case	OP_SQRT:
                    return(sqrt(left));
                default:
                    fprintf(stderr, "Unknown operator in EsCalcTree()\n");
                    assert(0);
                    return  (0.0);
            }
        }
    }

    assert(0);
    return 0;
}

static double _CalcEq(const std::vector<double>& x, const EqInfo& eq, const std::vector<EqInfo>& etab)
{
    double left, right;

    switch (eq._type) {
        case NODE_CONST:
            return(eq._val);
            break;

        case NODE_VAR: {
            int idx = eq._var;
            return x[idx];
            break;
        }

        case NODE_OPER: {
            left = _CalcEq(x, etab[eq._left], etab);
            right = _CalcEq(x, etab[eq._right], etab);
            switch (eq._op) {
                case	OP_PLUS:
                    return(left + right);
                case	OP_MINUS:
                    return(left - right);
                case	OP_UMINUS:
                    return(-right);
                case	OP_TIME:
                    return(left * right);
                case	OP_DIVIDE:
                    return(left / right);
                case	OP_SIN:
                    return(sin(left));
                case	OP_COS:
                    return(cos(left));
                case	OP_TG:
                    return(tan(left));
                case	OP_CTG:
                    return(1.0 / tan(left));
                case	OP_SEC:
                    return(1.0 / cos(left));
                case	OP_CSC:
                    return(1.0 / sin(left));
                case	OP_ARCSIN:
                    return(asin(left));
                case	OP_ARCCOS:
                    return(acos(left));
                case	OP_ARCTG:
                    return(atan(left));
                case	OP_ARCCTG:
                    return(atan(-left) + M_PI_2);
                case	OP_POW:
                    return(pow(left, right));
                case	OP_EEXP:
                    return(exp(left));
                case	OP_EXP:
                    return(exp(left * log(right)));
                case	OP_LN:
                    return(log(left));
                case	OP_LOG:
                    return(log(right) / log(left));
                case	OP_SQR:
                    return(left * left);
                case	OP_SQRT:
                    return(sqrt(left));
                default:
                    fprintf(stderr, "Unknown operator in EsCalcTree()\n");
                    assert(0);
                    return  (0.0);
            }
        }
    }

    assert(0);
    return 0;
}

static double _CalcObj(const std::vector<double>& x,
                       const std::vector<EqInfo>& eqs, int eqNum)
{
    std::vector<double> tmp;
    tmp.resize(eqNum);

    for (int i = 0; i < eqNum; i++) {
        //double v1 = _CalcEq(x, eqs[i], eqs);
        double v2 = _CalcEqNew1(x, eqs[i], eqs, i == eqNum - 1 ? -i : i, objEqHeads, objEqVals.size(), objEqVals);
        //assert(v1 == v2);
        tmp[i] = v2;
    }

    return _VecDot(tmp, tmp);
}

static void _CalcGrad(const std::vector<double>& x, std::vector<double>& g,
                      const std::vector<EqInfo>& eqs)
{
    int n = x.size();
    for (int i = 0; i < n; i++) {
        //double v1 = _CalcEq(x, eqs[i], eqs);
        double v2 = _CalcEqNew1(x, eqs[i], eqs, i == n - 1 ? -1 : i, gradEqHeads, gradEqVals.size(), gradEqVals);
        //assert(v1 == v2);
        g[i] = v2;
    }
}

static double _CalcObj(const std::vector<double>& x0, double h, const std::vector<double>& p,
                       const std::vector<EqInfo>& eqs, int eqNum)
{
    std::vector<double> xt;
    xt.resize(x0.size());
    _VecAxPy(x0, h, p, xt);
    return _CalcObj(xt, eqs, eqNum);
}

static void _FillVec(double *dev_v, double value, int n)
{
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    _FillVec_kernel<<<num_blocks, threadsPerBlock>>>(dev_v, value, n);
}

static void _InitH(double *dev_H, int n)
{
    int blocksPerRow = imin(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blocks(n, blocksPerRow);
    _Identity_kernel<<<blocks, threadsPerBlock>>>(dev_H, n);
}

static void _UpdateH(double *dev_H, const double *dev_s, const double *dev_Hy,
                     const double *dev_yTH, double sy, double tmp, int n)
{
    int blocksPerRow = imin(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blocks(n, blocksPerRow);
    _UpdateH_kernel<<<blocks, threadsPerBlock>>>(
            dev_H, dev_s, dev_Hy, dev_yTH, sy, tmp, n);
}

static void _CalcyTH(const double *dev_y, const double *dev_H, double *dev_yTH, int n)
{
    int blocksPerCol = imin(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blocks(n, blocksPerCol);
    _CalcyTH_kernel<<<blocks, threadsPerBlock>>>(dev_y, dev_H, dev_yTH, n);
}

static void _CalcHy(const double *dev_H, const double *dev_y, double *dev_Hy, int n)
{
    int blocksPerRow = imin(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blocks(n, blocksPerRow);
    _CalcHy_kernel<<<blocks, threadsPerBlock>>>(dev_H, dev_y, dev_Hy, n);
}

static void _Calcp(const double *dev_H, const double *dev_g, double *dev_p, int n)
{
    int blocksPerRow = imin(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blocks(n, blocksPerRow);
    _CalcHy_kernel<<<blocks, threadsPerBlock>>>(dev_H, dev_g, dev_p, n, true);
}

#define BFGS_MAXBOUND	1e+10
static void _DetermineInterval(
        const std::vector<double>& x0, double h, const std::vector<double>& p,
        double* left, double* right,
        const std::vector<EqInfo>& eqs, int eqNum)
{
    double	A, B, C, D, u, v, w, s, r;

    A = _CalcObj(x0, 0.0, p, eqs, eqNum);
    B = _CalcObj(x0, h, p, eqs, eqNum);
    if (B > A) {
        s = -h;
        C = _CalcObj(x0, s, p, eqs, eqNum);
        if (C > A) {
            *left = -h;
            *right = h;
            return;
        }
        B = C;
    }
    else {
        s = h;
    }
    u = 0.0;
    v = s;
    while (1) {
        s += s;
        if (fabs(s) > BFGS_MAXBOUND) {
            *left = *right = 0.0;
            return;
        }
        w = v + s;
        C = _CalcObj(x0, w, p, eqs, eqNum);
        if (C >= B)
            break;
        u = v;
        A = B;
        v = w;
        B = C;
    }
    r = (v + w) * 0.5;
    D = _CalcObj(x0, r, p, eqs, eqNum);
    if (s < 0.0) {
        if (D < B) {
            *left = w;
            *right = v;
        }
        else {
            *left = r;
            *right = u;
        }
    }
    else {
        if (D < B) {
            *left = v;
            *right = w;
        }
        else {
            *left = u;
            *right = r;
        }
    }
}

static void _GodenSep(
        const std::vector<double>& x0, const std::vector<double>& p,
        double left, double right, std::vector<double>& x,
        const std::vector<EqInfo>& eqs, int eqNum)
{
    static double	beta = 0.61803398874989484820;
    double			t1, t2, f1, f2;

    t2 = left + beta * (right - left);
    f2 = _CalcObj(x0, t2, p, eqs, eqNum);
    ENTRY1:
    t1 = left + right - t2;
    f1 = _CalcObj(x0, t1, p, eqs, eqNum);
    ENTRY2:
    if (fabs(t1 - t2) < epsZero2) {
        t1 = (t1 + t2) / 2.0;
        //printf("LineSearch t = %lf\n", t1*10000);

        _VecAxPy(x0, t1, p, x);
        return;
    }
    if ((fabs(left) > BFGS_MAXBOUND) || (fabs(left) > BFGS_MAXBOUND))
        return;
    if (f1 <= f2) {
        right = t2;
        t2 = t1;
        f2 = f1;
        goto ENTRY1;
    }
    else {
        left = t1;
        t1 = t2;
        f1 = f2;
        t2 = left + beta * (right - left);
        f2 = _CalcObj(x0, t2, p, eqs, eqNum);
        goto ENTRY2;
    }
}

static void _LinearSearch(
        const std::vector<double>& x0,
        const std::vector<double>& p,
        double h,
        std::vector<double>& x,
        const std::vector<EqInfo>& eqs,
        int eqNum)
{
    double	left, right;

    _DetermineInterval(x0, h, p, &left, &right, eqs, eqNum);
    if (left == right)
        return;

    //printf("%lf, %lf\n", left, right);
    _GodenSep(x0, p, left, right, x, eqs, eqNum);
}

#define	H_EPS1	1e-5
#define	H_EPS2	1e-5
#define	H_EPS3	1e-4

static bool _HTerminate(
        const std::vector<double>& xPrev,
        const std::vector<double>& xNow,
        double fPrev, double fNow,
        const std::vector<double>& gNow)
{
    double	ro;
    std::vector<double> xDif(xNow.size());

    if (_VecLen(gNow) >= H_EPS3)
        return false;

    _VecSub(xNow, xPrev, xDif);
    ro = _VecLen(xPrev);
    if (ro < H_EPS2)
        ro = 1.0;
    ro *= H_EPS1;
    if (_VecLen(xDif) >= ro)
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

void
AnalysisEqs(const std::vector<EqInfo>& eqTab, int eqNum, std::vector<int>& eqHeads)
{
    eqHeads.resize(eqNum);
    for (int i = 0; i < eqNum; i++) {
        const EqInfo& eq = eqTab[i];
        int left = eq._left;
        int right = eq._right;

        eqHeads[i] = left;
    }
}

int BFGSSolveEqs(char *data_path)
{
    double eps = _GetEps()*_GetEps();
    int itMax = _GetMaxIt();

    double step = _GetStep();

    std::vector<double> xNow, xKeep;
    std::vector<int> varMap, revMap;
    std::vector<EqInfo> objEqs;  // 目标函数
    int numObjEqs;
    std::vector<EqInfo> gradEqs;  // 目标函数导函数
    int numGradEqs;

    {
        FILE* fp = fopen(data_path, "rb");
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

    //Do optimization
    double fNow = 0, fPrev = 0;
    int n = xNow.size();
    int itCounter = 0;

    std::vector<double> gPrev, gNow, xPrev, p, y, s, yTH, Hy;  // p = H * g
    array2<double> H(n, n);

    xPrev = xNow;
    H.resize(n, n);
    gPrev.resize(n);
    gNow.resize(n);
    p.resize(n);
    y.resize(n);
    s.resize(n);
    yTH.resize(n);
    Hy.resize(n);

    double *dev_H, *dev_p, *dev_y, *dev_s, *dev_g, *dev_yTH, *dev_Hy;
    HANDLE_ERROR(cudaMalloc((void **) &dev_H, n * n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_p, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_y, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_s, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_g, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_yTH, n * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_Hy, n * sizeof(double)));

    float t_step6 = 0.0;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

//STEP1:
    fPrev = _CalcObj(xNow, objEqs, numObjEqs);
    _CalcGrad(xNow, gPrev, gradEqs);

    STEP2:
    _InitH(dev_H, n);
    for (int i = 0; i < n; i++) {
        p[i] = -gPrev[i];
    }
    _VecNorm(p);

    STEP3:
    if (itCounter++ > itMax)
        goto END;

    xPrev = xNow;
    _LinearSearch(xPrev, p, step, xNow, objEqs, numObjEqs);
    fNow = _CalcObj(xNow, objEqs, numObjEqs);
    std::cout << itCounter << " iterations, " << "f(x) = " << fNow << std::endl;

    if (fNow < eps)
        goto END;

    _CalcGrad(xNow, gNow, gradEqs);

//STEP4:
    if (_HTerminate(xPrev, xNow, fPrev, fNow, gNow))
        goto END;

//STEP5:
    if (fNow > fPrev) {
        _VecCopy(xNow, xPrev);
        goto STEP2;
    }

//STEP6:
    HANDLE_ERROR(cudaEventRecord(start, 0));
    _VecSub(gNow, gPrev, y);
    _VecSub(xNow, xPrev, s);

    {
        double sy = _VecDot(s, y);
        if (fabs(sy) < epsZero1) {
            HANDLE_ERROR(cudaEventRecord(stop, 0));
            HANDLE_ERROR(cudaEventSynchronize(stop));
            float elapsedTime;
            HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
            t_step6 += elapsedTime / 1000;
            goto END;
        }

        _FillVec(dev_yTH, 0.0, n);
        _FillVec(dev_Hy, 0.0, n);
        HANDLE_ERROR(cudaMemcpy(dev_y, y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        _CalcyTH(dev_y, dev_H, dev_yTH, n);
        _CalcHy(dev_H, dev_y, dev_Hy, n);
        HANDLE_ERROR(cudaDeviceSynchronize());

        double tmp = _VecDot(dev_yTH, dev_y, n);
        tmp = 1.0 + tmp / sy;
        HANDLE_ERROR(cudaMemcpy(dev_s, s.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_g, gNow.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        _UpdateH(dev_H, dev_s, dev_Hy, dev_yTH, sy, tmp, n);
        _Calcp(dev_H, dev_g, dev_p, n);
        _VecNorm(dev_p, n);
        HANDLE_ERROR(cudaMemcpy(p.data(), dev_p, n * sizeof(double), cudaMemcpyDeviceToHost));

        fPrev = fNow;
        _VecCopy(gPrev, gNow);
        _VecCopy(xPrev, xNow);

        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float elapsedTime;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        t_step6 += elapsedTime / 1000;

        goto STEP3;
    }

    END:
    std::cout << itCounter << " iterations" << std::endl;
    std::cout << "f(x) = " << fNow << std::endl;
    printf("### Step6 used %2.5f s ...\n", t_step6);

    //Put results back...
    if (fNow < eps) {
        printf("Solved!!!!\n");
        return true;
    }
    else {
        printf("Solver Failed!!!!\n");
        return false;
    }

    HANDLE_ERROR(cudaFree(dev_H));
    HANDLE_ERROR(cudaFree(dev_p));
    HANDLE_ERROR(cudaFree(dev_y));
    HANDLE_ERROR(cudaFree(dev_s));
    HANDLE_ERROR(cudaFree(dev_g));
    HANDLE_ERROR(cudaFree(dev_yTH));
    HANDLE_ERROR(cudaFree(dev_Hy));
}

int main()
{
    char data_path[] = "../../data/bfgs-large.dat";
    BFGSSolveEqs(data_path);
}

