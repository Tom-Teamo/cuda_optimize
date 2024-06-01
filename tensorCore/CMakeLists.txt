
add_executable(1_vec 1_vec.cu)
target_link_libraries(1_vec PRIVATE CUDA::cublas CUDA::cudart)

add_executable(2_bankC_co 2_bankC_co.cu)
target_link_libraries(2_bankC_co PRIVATE CUDA::cublas CUDA::cudart)

add_executable(3_async 3_async.cu)
target_link_libraries(3_async PRIVATE CUDA::cublas CUDA::cudart)

add_executable(4_reg 4_reg.cu)
target_link_libraries(4_reg PRIVATE CUDA::cublas CUDA::cudart)

add_executable(5_prefetch 5_prefetch.cu)
target_link_libraries(5_prefetch PRIVATE CUDA::cublas CUDA::cudart)

add_executable(6_prefetch_opti 6_prefetch_opti.cu)
target_link_libraries(6_prefetch_opti PRIVATE CUDA::cublas CUDA::cudart)

add_executable(7_reshape 7_reshape.cu)
target_link_libraries(7_reshape PRIVATE CUDA::cublas CUDA::cudart)

add_executable(8_swizzle 8_swizzle.cu)
target_link_libraries(8_swizzle PRIVATE CUDA::cublas CUDA::cudart)

add_executable(9_ldmatrix 9_ldmatrix.cu)
target_link_libraries(9_ldmatrix PRIVATE CUDA::cublas CUDA::cudart)

target_compile_options(9_ldmatrix PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

include_directories(/home/cwwu/cutlass/include/)
include_directories(/home/cwwu/cutlass/tools/util/include)

# 也可以直接sudo make install，这样cutlass就安装在/usr/local下了，无需再手动加载include路径
# 也可以在项目中添加third-party，设为submodule，只需要set(CUTLASS_INCLUDE_DIR ./third_party/cutlass/include)即可
# https://mp.weixin.qq.com/s/FXuFljYMc-8Zb8pHf--GPA
