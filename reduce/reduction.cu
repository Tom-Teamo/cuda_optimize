// 协作组是一种 CUDA 编程模型的特性，允许在同一个线程块中的线程之间进行协作。
// cooperative_groups 头文件提供了一些 API，使得在同一线程块内的线程能够更有效地进行协作。
// 这包括对线程块内线程索引、同步点、共享内存等的访问和操作。

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
                (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

namespace cg = cooperative_groups;

/*
	在CUDA编程中，要获取共享内存的指针，可以直接声明一个共享内存指针
	然后将其赋值为extern __shared__关键字后的数组、
	
	operator 用于重载解引用操作符（operator*）
	这是为了创建一个类型安全的包装器，以便在 CUDA 内核中访问共享内存。
*/
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};


__global__ void reduce_shuffle4(float* in, float* out)
{
    int tid = threadIdx.x;
    int offset_idx = (blockIdx.x * blockDim.x + threadIdx.x);
    float4* data = reinterpret_cast<float4*>(in);
    float sum = data[offset_idx].x + data[offset_idx].y + data[offset_idx].z + data[offset_idx].w;
    __syncthreads();

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    
    __shared__ float sdata[8];
    // int warp_idx = tid / 32;
    if (tid % 32 == 0) sdata[tid / 32] = sum;
    __syncthreads();

    if (tid < 8) {
        sum = sdata[tid];
        __syncthreads();
        // sum += __shfl_down_sync(0xffffffff, sum, 16);
        // sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xff000000, sum, 4);
        sum += __shfl_down_sync(0xff000000, sum, 2);
        sum += __shfl_down_sync(0xff000000, sum, 1);
    }

    if (tid == 0) out[blockIdx.x] = sum;
}

template <class T>
T reduceCPU(T *data, int size) {
  T sum = data[0];
  T c = (T)0.0;

  for (int i = 1; i < size; i++) {
    T y = data[i] - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

int main() {
	// array size
    const size_t size = 1024 * 1024 * 32;
	const int block_size = 256;
	const int grid_size = (size / 4 - 1) / block_size + 1;

	// allocate host input memory and init
    unsigned int bytes = size * sizeof(float);
    float *h_idata = (float *)malloc(bytes);
    for (long long i = 0; i < size; i++) {
        h_idata[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }

	// CPU result
	float output = reduceCPU<float>(h_idata, size);
	printf("CPU result = %.*f\n", 8, (double)output);

	// allocate device input and copy input data to device
	float *d_idata = NULL;
	checkCudaErrors(cudaMalloc(((void **)&d_idata), bytes));
	checkCudaErrors(
		cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice)
	);

	// allocate device output
	float* d_output = NULL;
	checkCudaErrors(cudaMalloc((void **)&d_output, grid_size * sizeof(float)));

	// allocate host output
	float* h_output = (float*)malloc(grid_size * sizeof(float));

	reduce_shuffle4<<<grid_size, block_size>>>(d_idata, d_output);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaMemcpy(h_output, d_output, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
	
	output = 0.f;
	for (int i = 0; i < grid_size; ++i) {
        output += h_output[i];
    }
	printf("GPU result = %.*f\n", 8, (double)output);

    return 0;
}

/*

	NVIDIA GeForce RTX 3090 的共享内存银行宽度是 256 字节。这是在 Ampere 架构中的一个常见值。

	----------------------------------------------------------

	Size of char: 1 bytes
	Size of short: 2 bytes
	Size of int: 4 bytes
	Size of unsigned int: 4 bytes
	Size of long: 8 bytes
	Size of long long: 8 bytes
	Size of float: 4 bytes
	Size of double: 8 bytes

	https://stackoverflow.com/questions/3935858
	和host 编译器相关

	----------------------------------------------------------
	########device query##########

	Device 0: "NVIDIA GeForce RTX 3090"
	CUDA Driver Version / Runtime Version          12.1 / 12.1
	CUDA Capability Major/Minor version number:    8.6
	Total amount of global memory:                 24238 MBytes (25414860800 bytes)
	(082) Multiprocessors, (128) CUDA Cores/MP:    10496 CUDA Cores
	GPU Max Clock rate:                            1740 MHz (1.74 GHz)
	Memory Clock rate:                             9751 Mhz
	Memory Bus Width:                              384-bit
	L2 Cache Size:                                 6291456 bytes
	Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
	Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
	Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
	Total amount of constant memory:               65536 bytes
	Total amount of shared memory per block:       49152 bytes(48 kb)
	Total shared memory per multiprocessor:        102400 bytes
	Total number of registers available per block: 65536
	Warp size:                                     32
	Maximum number of threads per multiprocessor:  1536
	Maximum number of threads per block:           1024
	Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
	Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
	Maximum memory pitch:                          2147483647 bytes
	Texture alignment:                             512 bytes
	Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
	Run time limit on kernels:                     Yes
	Integrated GPU sharing Host Memory:            No
	Support host page-locked memory mapping:       Yes
	Alignment requirement for Surfaces:            Yes
	Device has ECC support:                        Disabled
	Device supports Unified Addressing (UVA):      Yes
	Device supports Managed Memory:                Yes
	Device supports Compute Preemption:            Yes
	Supports Cooperative Kernel Launch:            Yes
	Supports MultiDevice Co-op Kernel Launch:      Yes
	Device PCI Domain ID / Bus ID / location ID:   0 / 4 / 0
	Compute Mode:
		< Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

	deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.1, CUDA Runtime Version = 12.1, NumDevs = 1
	Result = PASS

*/