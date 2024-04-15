
# add_library(cuda_comp SHARED main.cpp gemm_2.cu)
# set_target_properties(cuda_comp PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

#### https://cmake.org/cmake/help/latest/module/FindCUDA.html#commands
#### This is deprecated
# cuda_add_executable(gemm main.cpp gemm_4.cu)
# cuda_add_cublas_to_target(gemm -lcublas -lcurand)

# cuda_add_executable(gemm_cublas_v1 main.cpp MY_MUL_cublas_v1.cpp)
# cuda_add_cublas_to_target(gemm_cublas_v1 -lcublas -lcurand)

# add_executable(gemm_cublas_v2 main.cpp MY_MUL_cublas_v2.cpp)
# cuda_add_cublas_to_target(gemm_cublas_v2)


add_executable(gemm2 main.cpp gemm_2.cu)
target_link_libraries(gemm2 PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm3 main.cpp gemm_3.cu)
target_link_libraries(gemm3 PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm4 main.cpp gemm_4.cu)
target_link_libraries(gemm4 PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm4_1DBlockTile main.cpp gemm_4_block1Dtile.cu)
target_link_libraries(gemm4_1DBlockTile PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm4_2DBlockTile main.cpp gemm_4_block2Dtile.cu)
target_link_libraries(gemm4_2DBlockTile PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm_cublas_v1 main.cpp MY_MUL_cublas_v1.cpp)
target_link_libraries(gemm_cublas_v1 PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm_cublas_v2 main.cpp MY_MUL_cublas_v2.cpp)
target_link_libraries(gemm_cublas_v2 PRIVATE CUDA::cublas CUDA::cudart)