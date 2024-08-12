add_executable(transpose transpose.cu)
target_link_libraries(softmax PRIVATE CUDA::cublas CUDA::cudart)