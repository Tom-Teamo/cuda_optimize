add_executable(softmax softmax.cu main.cpp)
target_link_libraries(softmax PRIVATE CUDA::cublas CUDA::cudart)