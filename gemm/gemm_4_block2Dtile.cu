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

#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void tile_2d_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float val[TM][TN] = {0.f};

    int num_shared_block = CEIL_DIV(K, BK); // or CEIL_DIV(K, BN);
    A = &A[OFFSET(blockIdx.y * BM, 0, K)];
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    for (int i = 0; i < num_shared_block; ++i) {
        for (int m = 0; m < TM; ++m) {
            // A_row A_col 是相对于当前A的
            int A_row = threadIdx.y * TM + m;
            int A_col = threadIdx.x;
            // 需要计算全局范围（C矩阵的范围内）是否越界
            if ((blockIdx.y * BM + A_row) < M && (i * BK + A_col) < K) {
                As[A_row][A_col] = A[OFFSET(A_row, A_col, K)];
            } else {
                As[A_row][A_col] = 0.f;
            }
        }

        for (int n = 0; n < TN; ++n) {
            int B_row = threadIdx.y;
            int B_col = threadIdx.x * TN + n;
            
            if ((blockIdx.x * BN + B_col) < K && (i * BK + B_row) < N) {
                Bs[B_row][B_col] = B[OFFSET(B_row, B_col, N)];
            }
            else {
                Bs[B_row][B_col] = 0.f;
            }
        }
        
        __syncthreads();

        A += BK;
        B += BK * N;

        // for k 拿到里面性能下降严重
        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m) {
                int A_row = threadIdx.y * TM + m;
                for (int n = 0; n < TN; ++n) {
                    int B_col = threadIdx.x * TN + n;
                    val[m][n] += As[A_row][k] * Bs[k][B_col];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        int C_row = threadIdx.y * TM + m;
        for (int n = 0; n < TN; ++n) {
            int C_col = threadIdx.x * TN + n;
            if ((blockIdx.y * BM + C_row) < M && (blockIdx.x * BN + C_col) < N) {
                C[OFFSET(C_row, C_col, N)] = alpha * val[m][n] + beta * C[OFFSET(C_row, C_col, N)];
            }
        }
    }
}


void MY_MMult(cublasHandle_t handle, int M, int N, int K, float *A, int lda,
              float *B, int ldb, float *C, int ldc) {
    // 现在是什么情况呢
    // m n k 是要 整除 bm bn bk
    // BM 也必须整除 TM，BM TM是算法内部设计死的 所以只要代码里面满足就可以了 不像m n k 是用户输入的

    const int size = 16;
    const int tile_size = 8;
    const int BM = size * tile_size;
    const int BN = size * tile_size;
    const int BK = size;
    const int TM = tile_size;
    const int TN = tile_size;

    const int alpha = 1.0;
    const int beta = 0.0;

    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // note: change M and N here
    tile_2d_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}