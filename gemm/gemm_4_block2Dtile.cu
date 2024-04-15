#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <mma.h>


/*
在1维tile中，需要计算C矩阵中的Bc:[BLOCK, BLOCK]大小的数据，我们让每个线程读取[BLOCK, 1]大小的数据

现在，我们让每个线程读取[TM, TN]大小的数据
    在拷贝数据时，每个线程需要在 $A$ 矩阵中多拷贝 $TM$ 次：
    计算中间结果时，也需要多一层循环
*/
template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, float *b, float *c) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    float *a_begin = a + by * BLOCK * k;
    float *b_begin = b + bx * BLOCK;
    float *a_end = a_begin + k;

    float sum[BLOCK] = {0.f};

    __shared__ float a_smem[BLOCK][BLOCK];
    __shared__ float b_smem[BLOCK][BLOCK];

    for (float *a_ptr = a_begin, *b_ptr = b_begin; a_ptr < a_end;
            a_ptr += BLOCK, b_ptr += BLOCK * n) {

        // warp内的线程天然同步，因此tx写在后面更不容易发生bank conflict
        for (int i = 0; i < BLOCK; ++i) {
            a_smem[i][tx] = a_ptr[i * k + tx];
            b_smem[i][tx] = b_ptr[i * n + tx];
        }
     
        __syncthreads();

        for (int i = 0; i < BLOCK; i++) {
            // 下面是计算第i行 第tx列的结果
            for (int j = 0; j < BLOCK; j++) {
                sum[i] += a_smem[i][j] * b_smem[j][tx];
            }
        }
        
        __syncthreads();
    }

    for (int i = 0; i < BLOCK; i++){

        c[(by * BLOCK + i) * n + bx * BLOCK + tx] = sum[i];
    }
    
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {
    constexpr int BLOCK = 32;
    dim3 block(BLOCK, 1);
    dim3 grid((m + BLOCK - 1) / BLOCK , (n + BLOCK -1) / BLOCK );
    sgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, d_B, d_C);
}