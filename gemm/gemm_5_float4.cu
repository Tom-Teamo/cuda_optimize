#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <mma.h>


/*
    可以通过nsight compute看到，当前2D-tile还是stall在了memory，花在了访存等待上。
    我们可以进一步使用float4指令进一步优化程序的访存，让每个元素一次性读取4个浮点数，减少对内存的竞争。

    同时，我们在读取矩阵 $A$ 的内容时，还要对其进行转置，然后再存储到 shared memory 中，
    以方便后续线程计算时使用  float4  读取，以避免共享内存的 bank conflict。

    其实代码中的load_a_time可以消除掉，TM TN都是自己确定的，完全可以block中的所有线程一次性完成load
*/

#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void tile_2d_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    
    // 16 16
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    // 256
    const int thread_num = block_row_thread * block_col_thread;
    int num_shared_block = CEIL_DIV(K, BK);

    __shared__ float As[BK][BM];    // transpose shared A for avoid bank conflict
    __shared__ float Bs[BK][BN];

    float accum[TM][TN] = {0.};

    // 1 1
    // 这里刚好一个block里面的线程 每个线程执行一次float4指令 就可以加载完As Bs
    // 不然的话 还需要多次加载 这里的time指的就是需要加载几次
    const int load_a_cache_time = (BK * BM) / (thread_num * 4);  // Each thread load 4 float
    const int load_b_cache_time = (BK * BN) / (thread_num * 4);  // Each thread load 4 float

    A = &A[OFFSET(blockIdx.y * BM, 0, K)];
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    // thread id: [0,thread_num(256))
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    // 当前线程负责的As中的行和列（相对于A来说的）
    int a_tile_row = thread_id / (BK / 4);
    int a_tile_col = thread_id % (BK / 4) * 4;
    // 多次加载的时候移动的offset
    // 为每次读取单独开辟了空间存储，而不是复用之前的空间。
    // 如果复用之前的空间，会因为数据写入地址的依赖导致计算结果出错
    int a_tile_stride = BM / load_a_cache_time;

    int b_tile_row = thread_id / (BN / 4);
    int b_tile_col = thread_id % (BN / 4) * 4;
    int b_tile_stride = BK / load_b_cache_time;

    float As_cache[TM] = {0.};
    float Bs_cache[TN] = {0.};

    float4 tmp;

    #pragma unroll
    for (int i = 0; i < num_shared_block; ++i) {
        #pragma unroll
        for (int m = 0; m < BM; m += a_tile_stride) {
            // 为什么As需要transpose？ 原因不在于store阶段，而在于load也就是计算阶段
            // 在从global加载到sharedMem时候，As中行列对换就可以了
            tmp = FETCH_FLOAT4(A[OFFSET(a_tile_row + m, a_tile_col, K)]);
            As[a_tile_col][a_tile_row + m] = tmp.x;
            As[a_tile_col + 1][a_tile_row + m] = tmp.y;
            As[a_tile_col + 2][a_tile_row + m] = tmp.z;
            As[a_tile_col + 3][a_tile_row + m] = tmp.w;
        }
        #pragma unroll
        for (int k = 0; k < BK; k += b_tile_stride) {
            FETCH_FLOAT4(Bs[b_tile_row + k][b_tile_col]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + k, b_tile_col, N)]);
        }
        __syncthreads();
        A += BK;    // Start position of next tile block to be processed
        B += BK * N;    // Start position of next tile block to be processed

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; m += 4) {
                int A_row = threadIdx.y * TM + m;
                // 在计算index的时候，计算下标的时候还是常规的计算取到的As的行列
                // 在真正的从As中取数据的时候，对换行列即可
                FETCH_FLOAT4(As_cache[m]) = FETCH_FLOAT4(As[k][A_row]);
            }
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                int B_col = threadIdx.x * TN + n;
                FETCH_FLOAT4(Bs_cache[n]) = FETCH_FLOAT4(Bs[k][B_col]);
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += As_cache[m] * Bs_cache[n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int C_row = threadIdx.y * TM + m;
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int C_col = threadIdx.x * TN + n;
            tmp = FETCH_FLOAT4(C[OFFSET(C_row, C_col, N)]);
            tmp.x = alpha * accum[m][n] + beta * tmp.x;
            tmp.y = alpha * accum[m][n + 1] + beta * tmp.y;
            tmp.z = alpha * accum[m][n + 2] + beta * tmp.z;
            tmp.w = alpha * accum[m][n + 3] + beta * tmp.w;
            FETCH_FLOAT4(C[OFFSET(C_row, C_col, N)]) = FETCH_FLOAT4(tmp);
        }
    }
}


void MY_MMult(cublasHandle_t handle, int M, int N, int K, float *A, int lda,
              float *B, int ldb, float *C, int ldc) {

    const int size = 16;
    const int tile_size = 8;
    const int BM = size * tile_size;
    const int BN = size * tile_size;
    const int BK = 8;
    const int TM = tile_size;
    const int TN = tile_size;

    const int alpha = 1.0;
    const int beta = 0.0;

    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // note: change M and N here
    tile_2d_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


/*
block(size, size)
由于一个线程现在加载float4，确定一个block中的As(BM*BK)需要当前线程数目*4多少次
根据当前线程块中的线程tid，确定当前线程计算的As和Bs中的行和列。读取数据到As和Bs中。

*/