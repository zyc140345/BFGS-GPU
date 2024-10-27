//
// Created by 张易诚 on 24-10-12.
//

#include "util.h"

void RecordStartTime(cudaEvent_t start, cudaStream_t s) {
    CUDA_CHECK(cudaEventRecord(start, s));
}

float RecordStopTime(cudaEvent_t start, cudaEvent_t stop, cudaStream_t s) {
    CUDA_CHECK(cudaEventRecord(stop, s));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    return elapsedTime / 1000;
}

static const char *cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

void CublasCheck(cublasStatus_t err, const char *file, int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf("%s in %s at line %d\n", cublasGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

void CudaCheck(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}