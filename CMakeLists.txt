cmake_minimum_required(VERSION 3.16)

# this has to be set before project()
# NOT set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES "86")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(cuda_my_samples LANGUAGES CUDA CXX)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

# cuda include files
# 一旦 find_package 找到了软件包并成功配置，就可以使用相应的变量
# （例如 ${PackageName_INCLUDE_DIRS} 和 ${PackageName_LIBRARIES}）
# 来引用软件包的头文件路径和库文件路径。这些变量通常用于设置目标的依赖项和链接目标所需的库。
find_package(CUDAToolkit REQUIRED)

add_subdirectory(reduce)
add_subdirectory(gemm)
add_subdirectory(reduce_nvidia)
add_subdirectory(wmma)
add_subdirectory(float4)
add_subdirectory(flashAtten)
add_subdirectory(bank_conflict)