add_executable(bfgs_gpu
        main.cu kernel.h kernel.cu
        vec_mat_op.h vec_mat_op.cu
        util.h util.cu ../array2.h)
set_target_properties(bfgs_gpu PROPERTIES CUDA_ARCHITECTURES "86")  # double atomic add requires sm_60
add_compile_options(-G)  # for debugging
target_compile_options(bfgs_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)  # openmp
target_link_libraries(bfgs_gpu -lcublas -lgomp)
target_link_libraries(bfgs_gpu -lcublas)