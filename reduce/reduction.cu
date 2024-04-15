// 协作组是一种 CUDA 编程模型的特性，允许在同一个线程块中的线程之间进行协作。
// cooperative_groups 头文件提供了一些 API，使得在同一线程块内的线程能够更有效地进行协作。
// 这包括对线程块内线程索引、同步点、共享内存等的访问和操作。

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef MIN
#define MIN(x, y) ((x < y) ? x : y)
#endif

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
                (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}


////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction
// kernel For the kernels >= 3, we set threads / block to the minimum of
// maxThreads and n/2. For kernels < 3, we set to the minimum of maxThreads and
// n.  For kernel 6, we observe the maximum specified number of blocks, because
// each thread in that kernel can process a variable number of elements.
///////////////////////////////////////////////////////////`1/////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));

  if (whichKernel < 3) {
    threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;
  } else {
    threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
  }

  if ((float)threads * blocks >
      (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    printf("n is too large, please choose a smaller number!\n");
  }

  if (blocks > prop.maxGridSize[0]) {
    printf(
        "Grid size <%d> exceeds the device capability <%d>, set block size as "
        "%d (original %d)\n",
        blocks, prop.maxGridSize[0], threads * 2, threads);

    blocks /= 2;
    threads *= 2;
  }

  if (whichKernel >= 6) {
    blocks = MIN(maxBlocks, blocks);
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

__global__ void reduce0(int *d_idata, int *d_odata, int size) {
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	int *sdata = SharedMemory<int>();

	sdata[tid] = (i < size)? d_idata[i] : 0;

	cg::sync(cta);

	// do reduction in shared mem
	for (unsigned int i = 1; i < blockDim.x; i *= 2) {
		if (tid % (2 * i) == 0) {
			sdata[tid] += sdata[tid + i];
		}
		cg::sync(cta);
	}

	// write result for this block to global mem
	if (tid == 0) {
		d_odata[blockIdx.x] = sdata[0];
	}
}

/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
__global__ void reduce1(int *d_idata, int *d_odata, int size) {
	// handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	int *sdata = SharedMemory<int>();

	//load mem data
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = i < size? d_idata[i]: 0;
	cta.sync();

	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;

		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}

		cg::sync(cta);
	}

	// write result for this block to global mem
	if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
__global__ void reduce2(int *d_idata, int *d_odata, int size) {
	// handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	int *sdata = SharedMemory<int>();

	// load mem data
	unsigned int tid = threadIdx.x;
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	sdata[tid] = i < size? d_idata[i]: 0;
	cta.sync();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		cta.sync();
	}

	if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
__global__ void reduce3(int *d_idata, int *d_odata, int size) {
	// handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	int *sdata = SharedMemory<int>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	int my_sum = i < size? d_idata[i]: 0;
	if (i + blockDim.x < size)	my_sum += d_idata[i + blockDim.x];
	sdata[tid] = my_sum;

	cta.sync();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		cta.sync();
	}

	if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses the warp shuffle operation if available to reduce
    warp synchronization. When shuffle is not available the final warp's
    worth of work is unrolled to reduce looping overhead.

    See
   http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    for additional information about using shuffle to perform a reduction
    within a warp.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <unsigned int blockSize> 
__global__ void reduce4(int *d_idata, int *d_odata, int size) {
	// handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	int *sdata = SharedMemory<int>();

	// index
	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * (blockDim.x *2) + threadIdx.x;

	// load data
	int sum = id < size? d_idata[id]: 0;
	if (id + blockSize < size) sum += d_idata[id + blockSize];
	sdata[tid] = sum;
	cta.sync();

	// reduce
	for (unsigned int i = blockDim.x / 2; i > 32; i >>= 1) {
		if (tid < i) {
			sdata[tid] = sum = sum + sdata[tid + i];
		}
		cta.sync();
	}

	// if (tid == 0 && blockIdx.x == 0) printf("---%d---", blockDim.x);
	// unroll the last loops
	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
	if (cta.thread_rank() < 32) {
		if (blockDim.x >= 64 ) sum += sdata[tid + 32];
		// reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
			sum += tile32.shfl_down(sum, offset);
		}
	}

	// write result for this block to global mem
  	if (cta.thread_rank() == 0) d_odata[blockIdx.x] = sum;
}

template <unsigned int block_size>
__global__ void reduce5(int *d_idata, int *d_odata, int size) {
	// handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	// shared memory
	int *sdata = SharedMemory<int>();

	// index
	unsigned int tid = threadIdx.x;
	unsigned int id = threadIdx.x + blockIdx.x * (block_size * 2);

	// transfer data from global to shared memory
	int sum = id < size? d_idata[id]: 0;
	if (id + block_size < size) sum += d_idata[id + block_size];
	sdata[tid] = sum;
	
	cta.sync();

	// reduce
	if (block_size >= 512 && tid < 256) {
		sdata[tid] = sum = sum + sdata[tid + 256];
	}
	cta.sync();

	if (block_size >= 256 && tid < 128) {
		sdata[tid] = sum = sum + sdata[tid + 128];
	}
	cta.sync();

	if ((block_size >= 128) && (tid < 64)) {
		sdata[tid] = sum = sum + sdata[tid + 64];
	}
  	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32) {
		if (block_size >= 64) sum += sdata[tid + 32];
		// reduce final warp
		for (int i = tile32.size() / 2; i > 0; i /= 2) {
			sum += tile32.shfl_down(sum, i);
		}
	}

	if (cta.thread_rank() == 0) d_odata[blockIdx.x] = sum;
}

void reduce(int size, int thread_size, int block_size,
			int which_kenrnel, int *d_idata, 
			int *d_odata) {
	
	dim3 dim_block(thread_size, 1, 1);
	dim3 dim_grid(block_size, 1, 1);

	// shared memory size
	int smem_size =
	(thread_size <= 32) ? 2 * thread_size * sizeof(int) : thread_size * sizeof(int);

	switch (which_kenrnel) {
		case 0:
			reduce0<<<dim_grid, dim_block, smem_size>>>(d_idata, d_odata, size);
			break;
		case 1:
			reduce1<<<dim_grid, dim_block, smem_size>>>(d_idata, d_odata, size);
			break;
		case 2:
			reduce2<<<dim_grid, dim_block, smem_size>>>(d_idata, d_odata, size);
			break;
		case 3:
			reduce3<<<dim_grid, dim_block, smem_size>>>(d_idata, d_odata, size);
			break;
		case 4:
			switch (thread_size) {
				case 512:
					reduce4<512>
						<<<dim_grid, dim_block, smem_size>>>(d_idata, d_odata, size);
					break;

				case 256:
					reduce4<256>
						<<<dim_grid, dim_block, smem_size>>>(d_idata, d_odata, size);
					break;
			}
		case 5:
			reduce5<256>
				<<<dim_grid, dim_block, smem_size>>>(d_idata, d_odata, size);
			break;
	}

}

int run_test(int *d_idata, int size, int which_kenrnel, 
			  int test_iterations, int max_blocks, int max_threads) {
	
	int num_threads = 0, num_blocks = 0;
	getNumBlocksAndThreads(which_kenrnel, size, max_blocks, max_threads, num_blocks, num_threads);

	int *h_odata = (int *)malloc(num_blocks * sizeof(int));

	int *d_odata;
	checkCudaErrors(cudaMalloc((void **)&d_odata, num_blocks * sizeof(int)));

	int *d_intermediate;
	checkCudaErrors(
		cudaMalloc((void **)&d_intermediate, num_blocks * sizeof(int))
	);

	int gpu_result = 0;
	bool need_read_back = true;
	

	for(int i = 0; i < test_iterations; i++) {
		gpu_result = 0;
		cudaDeviceSynchronize();

		reduce(size, num_threads, num_blocks, which_kenrnel, d_idata, d_odata);
		// check whether the kernel execution generated errors
		checkCudaErrors(cudaGetLastError());

		// continue to reduce the blocks on the GPU
		int s = num_blocks;
		int kernel = which_kenrnel;

		while (s > 1) {
			int threads = 0, blocks = 0;
			getNumBlocksAndThreads(kernel, s, max_blocks, max_threads, blocks, threads);
			checkCudaErrors(cudaMemcpy(d_intermediate, d_odata, s * sizeof(int), cudaMemcpyDeviceToDevice));

			reduce(s, threads, blocks, kernel, d_intermediate, d_odata);

			if (kernel < 3) {
				s = (s + threads - 1) / threads;
			} else {
				s = (s + (threads * 2 - 1)) / (threads * 2);
			}
		}

		if (s > 1) {
			// copy result from device to host
			checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(int),
									cudaMemcpyDeviceToHost));

			for (int i = 0; i < s; i++) {
				gpu_result += h_odata[i];
			}

			need_read_back = false;
      	}
	}


	if (need_read_back) {
		// copy final sum from device to host
		checkCudaErrors(
			cudaMemcpy(&gpu_result, d_odata, sizeof(int), cudaMemcpyDeviceToHost));
	}
	checkCudaErrors(cudaFree(d_intermediate));

	return gpu_result;

}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
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
    int size = 1 << 24;
	// threads and blocks used in kernel launch
    int max_threads = 256;
	int max_blocks = 64;
	// which kernel to run
    int which_kenrnel = 4;
	// number of test iterations, compute the average performance
	int test_iterations = 1;


    printf("size: %d, which_kenrnel: %d\n", size, which_kenrnel);
    printf("max_threads: %d\n", max_threads);

	// allocate host input memory
    unsigned int bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    for (int i = 0; i < size; i++) {
        // Keep the numbers small so we don't get truncation error in the sum
        h_idata[i] = (int)(rand() & 0xFF);   
    }

	int num_blocks = 0;
	int num_threads = 0;
	getNumBlocksAndThreads(which_kenrnel, size, max_blocks, max_threads,
							num_blocks, num_threads);

	// allocate device memory
	int *d_idata = NULL;
	checkCudaErrors(cudaMalloc(((void **)&d_idata), bytes));
	// copy input data to device
	checkCudaErrors(
		cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice)
	);
	int gpu_result = 0;
	gpu_result = run_test(d_idata, size, which_kenrnel, test_iterations, num_blocks, num_threads);

	int cpu_result = reduceCPU<int>(h_idata, size);

	printf("\nGPU result = %d\n", (int)gpu_result);
	printf("CPU result = %d\n\n", (int)cpu_result);

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