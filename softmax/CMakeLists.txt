add_executable(softmax softmax.cu)
target_link_libraries(softmax PRIVATE CUDA::cublas CUDA::cudart)