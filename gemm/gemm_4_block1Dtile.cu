#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <mma.h>

/*
    各个compute capability下的技术细节：
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability

    SM内最大的常驻blocks：16
    SM内最大的常驻warp：48
    SM内最大的常驻thread：1536

    32-bit的寄存器数量：64k（k=1024）
    block内最多的32-bit寄存器数量：64k
    thread内最多的32-bit寄存器数量：255

    SM内最大的sMem：100KB
    block内最大的sMem：99KB
*/

/*
这要求M是BM的倍数，N是BN的倍数，K是BK的倍数

什么是 1维 tile？
    之前，需要计算C矩阵中的Bc:[BLOCK, BLOCK]大小的数据（因为矩阵大小未知 sharedMem有限 因此也需要分块）
        block中是需要 BLOCK* BLOCK 数量的线程的，每个线程计算Bc中的一个元素
    
    现在，需要计算C矩阵中的Bc:[BLOCK, BLOCK]大小的数据
        我们让每个线程读取[BLOCK, 1]大小的数据
        每个线程同样计算 [BLOCK, 1]大小的Bc的数据（下列代码每个线程计算Bc中的一列）
*/


#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <const int BM, const int BN, const int BK, const int TM>
__global__ void tile_1d_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float val[TM] = {0.};
    int num_shared_block = CEIL_DIV(K, BK); // or CEIL_DIV(K, BN);
    // 而&运算符被用来获取该区域的地址
    A = &A[OFFSET(blockIdx.y * BM, 0, K)];
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    for (int i = 0; i < num_shared_block; ++i) {
        // Copy data from global memory to shared memory
        for (int m = 0; m < TM; ++m) {
            int A_row = threadIdx.y * TM + m;
            int A_col = threadIdx.x;
            if ((blockIdx.y * BM + A_row) < M && (i * BK + A_col) < K) {
                As[A_row][A_col] = A[OFFSET(A_row, A_col, K)];
            } else {
                As[A_row][A_col] = 0.;
            }
        }
        int B_row = threadIdx.y;
        int B_col = threadIdx.x;
        if ((i * BK + B_row) < K && (blockIdx.x * BN + B_col) < N) {
            Bs[B_row][B_col] = B[OFFSET(B_row, B_col, N)];
        } else {
            Bs[B_row][B_col] = 0.;
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m) { 
                int A_row = threadIdx.y * TM + m;
                int B_col = threadIdx.x;
                val[m] += As[A_row][k] * Bs[k][B_col];
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        int C_row = threadIdx.y * TM + m;
        int C_col = threadIdx.x;
        if ((blockIdx.y * BM + C_row) < M && (blockIdx.x * BN + C_col) < N) {
            C[OFFSET(C_row, C_col, N)] = alpha * val[m] + beta * C[OFFSET(C_row, C_col, N)];
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
    const int BN = size;
    const int BK = size;
    const int TM = tile_size;

    const int alpha = 1.0;
    const int beta = 0.0;

    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // note: change M and N here
    tile_1d_kernel<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}