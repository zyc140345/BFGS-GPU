cmake_minimum_required(VERSION 3.22.1)
project(bfgs_gpu)

set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(bfgs_cpu array2.h bfgs_cpu.cpp)

add_subdirectory(gpu)
