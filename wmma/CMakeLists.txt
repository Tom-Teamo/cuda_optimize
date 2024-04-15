add_executable(wmma wmma.cu)
target_link_libraries(wmma PRIVATE CUDA::cublas CUDA::cudart)