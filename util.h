//
// Created by 张易诚 on 24-10-9.
//

#ifndef BFGS_GPU_UTIL_H
#define BFGS_GPU_UTIL_H

#include <cstdio>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define imin(a,b) (a<b?a:b)

#endif //BFGS_GPU_UTIL_H
