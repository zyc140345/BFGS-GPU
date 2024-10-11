add_compile_options(-G)  # 符号表
add_executable(bfgs_gpu
        main.cu kernel.h kernel.cu
        util.h ../array2.h)
set_target_properties(bfgs_gpu PROPERTIES CUDA_ARCHITECTURES "86")