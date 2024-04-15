
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

__global__ void reduce(int* d_in_data, int *d_out_data, int size) {
	auto cta = cg::this_thread_block();
	int* s_mem = SharedMemory<int>();

	int t_id = threadIdx.x;
	int id = blockIdx.x * (blockDim.x * 2) + t_id;

	int sum = id < size? d_in_data[id]: 0;
	if (id + blockDim.x < size) sum += d_in_data[id + blockDim.x];
	s_mem[t_id] = sum;

	cta.sync();

	for (int i = blockDim.x / 2; i > 32; i >>= 1) {
		if (t_id < i) {
			s_mem[t_id] = sum = sum + s_mem[t_id + i];
		}
		cta.sync();
	}

	auto tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32) {
		if (blockDim.x >= 64) sum += s_mem[t_id + 32];
		for (int offset = tile32.size() /2 ; offset > 0; offset /= 2) {
			sum += tile32.shfl_down(sum, offset);
		}
	}

	if (cta.thread_rank() == 0) d_out_data[blockIdx.x] = sum;
}














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