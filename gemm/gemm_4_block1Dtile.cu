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

    TODO:
        将各个参数写成变量！
*/

// template <int BM, int BN, int BK, int TM>
// __global__ void sgemm(int m, int n, int k, float *a, float *b, float *c) {

//     int tx = threadIdx.x;
//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     float *a_begin = a + by * BM * k;
//     float *b_begin = b + bx * BN;
//     float *a_end = a_begin + k;

//     // each thread computes the [TM] elements in C
//     float sum[TM] = {0.f};

//     __shared__ float a_smem[BM][BK];
//     __shared__ float b_smem[BK][BN];

//     for (float *a_ptr = a_begin, *b_ptr = b_begin; a_ptr < a_end;
//             a_ptr += BK, b_ptr += BK * n) {

//         // warp内的线程天然同步，因此tx写在后面更不容易发生bank conflict
//         for (int i = 0; i < BM; ++i) {
//             a_smem[i][tx] = a_ptr[i * k + tx];
//         }

//         for (int i = 0; i < BN; ++i) {
//             b_smem[tx][i] = b_ptr[tx * n + i];
//         }
        
//         __syncthreads();

//         for (int i = 0; i < BM; i++) {
//             // 下面是计算第i行 第tx列的结果
//             for (int j = 0; j < BN; j++) {
//                 sum[i] += a_smem[i][j] * b_smem[j][tx];
//             }
//         }
//         __syncthreads();
//     }

//     for (int i = 0; i < BM; i++){
//         c[(by * BM + i) * n + bx * BM + tx] = sum[i];
//     }
    
// }


// void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
//               float *d_B, int ldb, float *d_C, int ldc) {
//     // 现在是什么情况呢
//     // m n k 是要 整除 bm bk 
//     const int BM = 32;
//     const int BN = 32;
//     const int BK = 32;

//     // assume that BM % TM == 0
//     const int TM = 32;

//     dim3 block(BM, 1);
//     //assume that m % BM == 0
//     dim3 grid(m / BM , n / BN );
//     sgemm<BM, BN, BK, TM><<<grid, block>>>(m, n, k, d_A, d_B, d_C);
// }


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