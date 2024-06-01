
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

add_executable(gemm5_float4 main.cpp gemm_5_float4.cu)
target_link_libraries(gemm5_float4 PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm6_double_buffer main.cpp gemm6_double_buffer.cu)
target_link_libraries(gemm6_double_buffer PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm7_no_bankConf main.cpp gemm7_no_bankConf.cu)
target_link_libraries(gemm7_no_bankConf PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm7_no_BC_z_order main.cpp gemm7_no_BC_z_order.cu)
target_link_libraries(gemm7_no_BC_z_order PRIVATE CUDA::cublas CUDA::cudart)

add_executable(z_lishaoxia_gemm z_lishaoxia_gemm.cu)
target_link_libraries(z_lishaoxia_gemm PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm_cublas_v1 main.cpp MY_MUL_cublas_v1.cpp)
target_link_libraries(gemm_cublas_v1 PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm_cublas_v2 main.cpp MY_MUL_cublas_v2.cpp)
target_link_libraries(gemm_cublas_v2 PRIVATE CUDA::cublas CUDA::cudart)