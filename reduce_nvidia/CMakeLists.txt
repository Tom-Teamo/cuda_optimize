add_executable(reduce_nvidia reduction.cpp reduction_kernel.cu)
include_directories(../utils)
target_link_libraries(reduce_nvidia PRIVATE CUDA::cudart)