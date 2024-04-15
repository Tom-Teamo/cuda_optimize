# Warp Matrix Operations

This is the usage of the warp matrix operations with the warp matrix operations in C++. See [wmma](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions).

> C++ warp matrix operations leverage Tensor Cores to accelerate matrix problems of the form D=A*B+C. These operations are supported on mixed-precision floating point data for devices of compute capability 7.0 or higher. This requires co-operation from all threads in a warp. In addition, these operations are allowed in conditional code only if the condition evaluates identically across the entire warp, otherwise the code execution is likely to hang.

注意：该实例进行了简单的TensorCore的MMA，并没有特别的详细的进行优化，如果对TensorCore的优化感兴趣，可以参考resorces中的#2链接。

与FP32 Core类似，Tensor Core就是一个运算单元，前者输入两个浮点数，返回一个浮点数加法结果，后者输入两个矩阵，返回矩阵乘法结果。在cuda C的tensor core接口（wmma）中，kernel核函数中一次tensor core的运算需要占用一个warp的线程（32个）。由于tensor core的一次运算的矩阵大小是固定的，所需线程数也是固定的，所以我们多个tensor core并行运算只需要对矩阵、线程进行分割即可，下面讲讲怎么分割。

假设A[M, K], B[K, N]，其中M是A的行数，K是A的列数，N是B的列数，tensor core的一次矩阵运算的形状为[m, k] * [k, n] = [m, n]，其中从A矩阵中分割出[m, k]的子矩阵，从B矩阵分割出[k, n]的子矩阵，得到一个[m, n]的子矩阵。通过简单的计算可得，A矩阵要求在y方向上需要M / m个warp的线程（每个warp负责[m, k]的矩阵），B矩阵要求在x方向上需要N / n个warp的线程，而在kernel内进行K / k次的循环累加即可得到C中[m, n]的子矩阵。

由于tensor core不接受float的输入，所以使用半精度half(FP16)作为输入，float作为输出

in the `wmma.cu` file, I use the thrust library to create the host and device matrix. For more infomation, see [thrust](https://docs.nvidia.com/cuda/thrust/index.html) and [thrust pdf](https://docs.nvidia.com/cuda/pdf/Thrust_Quick_Start_Guide.pdf).

## resources

> https://forums.developer.nvidia.com/t/wmma-what-does-warp-matrix-operations-mean/229732/4
> https://blog.csdn.net/jianjianheng/article/details/119953806
> https://github.com/c3sr/tcu_scope/tree/master