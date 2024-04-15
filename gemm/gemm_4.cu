#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <mma.h>

/*
Note    
    About the bank conflict: 
        https://blog.51cto.com/u_15054042/4102569
        https://stackoverflow.com/questions/78080926
        32 个bank 每个32bit
        多个线程读取同一个32-word并不会产生bank conflict
*/

/*
    上一个版本中，将global memory中的数据加载到shared memory中，让访存的代价从几百个cycle降低到几十个cycle。
    但是这并不改变本质问题，一次乘累加只需要几个cycle
    问题的关键在于核心代码的循环计算部分，计算访存比过低，最终导致访存延迟不能被隐藏，从而性能不理想。

    从全局内存加载到shared mem，是需要两次迭代的，因为一个线程需要负责STRIDE * STRIDE的数据加载，
    比如STRIDE=2 那么每个thread需要负责4个数据的加载

    
*/



template <int BLOCK, int STRIDE>
__global__ void sgemm(int m, int n, int k, float *a, float *b, float *c) {

    constexpr int STEP = BLOCK * STRIDE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    float *a_begin = a + by * STEP * k;
    float *b_begin = b + bx * STEP;
    float *a_end = a_begin + k;

    // for **each thread**, it calculates for results.
    float sum[STRIDE][STRIDE] = {0.f};
    __shared__ float a_smem[STEP][STEP];
    __shared__ float b_smem[STEP][STEP];

    for (float *a_ptr = a_begin, *b_ptr = b_begin; a_ptr < a_end;
            a_ptr += STEP, b_ptr += STEP * n) {

        /*
            generate 16-way bank conflict for shared memory storing
            in a warp, it store data in column-major style, and s_mem'size is (32,32), 4 bytes each data
            So in column major-style, it stores at (0,0) (1,0)...(15,0)     (0,1) (1,1)...(15,1)
                                                   |-----conflict-----|     |-----conflict-----|
            About 16-way conflict, So it performs poorly
        */
        // for (int i = 0; i < STRIDE; i++) { // i_th row
        //     for (int j = 0; j < STRIDE; j++) { // j_th column
        //         a_smem[i * BLOCK + tx][j * BLOCK + ty] = 
        //             a_ptr[(i * BLOCK + tx) * k + j * BLOCK + ty];
        //         b_smem[i * BLOCK + tx][j * BLOCK + ty] = 
        //             b_ptr[(i * BLOCK + tx) * k + j * BLOCK + ty];
        //     }
        // }

        /*  from how-to-optimize-gemm
            this causes about 2-way bank conflict when BLOCK=16 and STRIDE=2
            this causes about 4-way bank conflict when BLOCK=16 and STRIDE=4
        */
        for (int i = 0; i < STRIDE; ++i) {
            for (int j = 0; j < STRIDE; ++j) {
                a_smem[ty * STRIDE + i][tx * STRIDE + j] =
                    a_ptr[(ty * STRIDE + i) * k + tx * STRIDE + j];
                b_smem[ty * STRIDE + i][tx * STRIDE + j] =
                    b_ptr[(ty * STRIDE + i) * n + tx * STRIDE + j];
            }
        }


        /*
            there is 2-way bank conflict when BLOCK=16 STRIDE=2
            there is 2-way bank conflict when BLOCK=16 STRIDE=4
        */
        // for (int i = 0; i < STRIDE; i++) { // i_th row
        //     for (int j = 0; j < STRIDE; j++) { // j_th column
        //         a_smem[i * BLOCK + ty][j * BLOCK + tx] = 
        //             a_ptr[(i * BLOCK + ty) * k + j * BLOCK + tx];
        //         b_smem[i * BLOCK + ty][j * BLOCK + tx] = 
        //             b_ptr[(i * BLOCK + ty) * k + j * BLOCK + tx];
        //     }
        // }

        // int medium = blockDim.x / STRIDE;
        // int row_a = ty * BLOCK * BLOCK
        // for (int i = 0; i < STRIDE * STRIDE; i++)
        // {
            
        //     a_smem[][] = a[];
        //     b_smem[][] = b[];
        // }
        
        __syncthreads();


        // for (int i = 0; i < STRIDE; i++) {
        //     for (int j = 0; j < STRIDE; j++) {
        //         for (int kk = 0; kk < STEP; kk++) {
        //             sum[i][j] += a_smem[i * BLOCK + tx][kk] * b_smem[kk][j * BLOCK + ty];
        //         }
        //     }
        // }

        #pragma unroll
        for (int i = 0; i < STRIDE; ++i) {
            #pragma unroll
            for (int j = 0; j < STRIDE; ++j) {
                #pragma unroll
                for (int kk = 0; kk < STEP; ++kk) {
                    sum[i][j] += 
                        a_smem[ty * STRIDE + i][kk] * b_smem[kk][tx * STRIDE + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < STRIDE; ++i) {
        #pragma unroll
        for (int j = 0; j < STRIDE; ++j) {
            c[(STEP * by + ty * STRIDE + i) * n + STEP * bx + tx * STRIDE + j] = 
                sum[i][j];
        }
    }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {
    constexpr int BLOCK = 16;
    constexpr int STRIDE = 4;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK -1) / BLOCK / STRIDE);
    sgemm<BLOCK, STRIDE><<<grid, block>>>(m, n, k, d_A, d_B, d_C);
}