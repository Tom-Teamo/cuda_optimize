# 添加libtorch进入prefix_path，从而能够被
# find_package(), find_program(), find_library(), find_file(), 和 find_path()
# 函数找到。每个命令都将按照自己的文档中的指定添加适当的子目录(如 bin lib include)
list(APPEND CMAKE_PREFIX_PATH "/usr/local/include/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")