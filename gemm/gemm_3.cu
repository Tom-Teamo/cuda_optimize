#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>

namespace cg = cooperative_groups;

/*
Note
    ncu --metrics smsp__sass_average_data_bytes_per_wavefront_mem_shared gemm
    其中，shared_efficiency=shared_efficiency等于smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct
    
*/

/*
    Each thread block is responsible for computing one square sub-matrix Csub of C and 
    each thread within the block is responsible for computing one element of Csub.
*/
template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, float *b, float *c)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    float *begin_a = a + bx * BLOCK * k;
    float *begin_b = b + by * BLOCK;
    float *end_a = begin_a + k;

    float sum = 0.f;

    cg::thread_block cta = cg::this_thread_block();

    // init the shared memory
    __shared__ float a_smem[BLOCK][BLOCK];
    __shared__ float b_smem[BLOCK][BLOCK];

    /* good */
    for(float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
        a_ptr += BLOCK, b_ptr += BLOCK * n) {
        
        // note: exchange tx and ty cause the performance decline by half
        a_smem[ty][tx] = a_ptr[ty * k + tx];
        b_smem[ty][tx] = b_ptr[ty * n + tx];
        __syncthreads();

#pragma unroll  // has little impact
        for (int i = 0; i < BLOCK; i++) {
            sum += a_smem[ty][i] * b_smem[i][tx];
        }
        __syncthreads();
    }

    c[(BLOCK * bx + ty) * n + BLOCK * by + tx] = sum;
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

    constexpr int BLOCK = 32;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

    sgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, d_B, d_C);
}