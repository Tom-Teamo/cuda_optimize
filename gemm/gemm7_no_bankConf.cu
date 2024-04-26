#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <mma.h>


#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void no_share_conflict_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;
    const int THREAD_TILE = TM / 4;
    // The left and uppermost element position of thread tile in block
    int start_col = blockIdx.x * BN;
    int start_row = blockIdx.y * BM;
    int tx = threadIdx.x * TN;
    int ty = threadIdx.y * TM;

    __shared__ float As[2][BK][BM];    // transpose shared A for avoid bank conflict, for double buffering
    __shared__ float Bs[2][BK][BN];    // for double buffering

    float accum[TM][TN] = {0.};

    const int load_a_cache_time = (BK * BM) / thread_num / 4;  // Each thread load 4 float
    const int load_b_cache_time = (BK * BN) / thread_num / 4;  // Each thread load 4 float

//    float load_a_cache[4 * load_a_cache_time];
    float load_a_cache[4];
//    float load_b_cache[4 * load_b_cache_time];

    A = &A[OFFSET(start_row, 0, K)]; // Set block start position
    B = &B[OFFSET(0, start_col, N)];
    C = &C[OFFSET(start_row, start_col, N)];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int a_tile_row = thread_id / (BK / 4);
    int a_tile_col = thread_id % (BK / 4) * 4;
    int a_tile_stride = BM / load_a_cache_time;
//    printf("A tile row, col, stride %d, %d, %d", a_tile_row, a_tile_col, a_tile_stride);

    int b_tile_row = thread_id / (BN / 4);
    int b_tile_col = thread_id % (BN / 4) * 4;
    int b_tile_stride = BK / load_b_cache_time;

    float a_reg[2][TM] = {0.};  // double buffering
    float b_reg[2][TN] = {0.};  // double buffering

    int write_idx = 0;

#pragma unroll
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int cache_idx = i / a_tile_stride * 4;
            FETCH_FLOAT4(load_a_cache) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            // Use load_a_cache for load 4 float at a time
            // As is saved as transpose matrix
            As[write_idx][a_tile_col][a_tile_row + i] = load_a_cache[cache_idx];
            // 这里 stride = 128，有 shared memory bank 冲突
            As[write_idx][a_tile_col + 1][a_tile_row + i] = load_a_cache[cache_idx + 1];
            As[write_idx][a_tile_col + 2][a_tile_row + i] = load_a_cache[cache_idx + 2];
            As[write_idx][a_tile_col + 3][a_tile_row + i] = load_a_cache[cache_idx + 3];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(Bs[write_idx][b_tile_row + i][b_tile_col]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
        }
        __syncthreads();
        A += BK;    // Start position of next tile block to be processed
        B += BK * N;    // Start position of next tile block to be processed

#pragma unroll
        for (int i = 0; i < BK; ++i) {
#pragma unroll
            for (int t = 0; t < THREAD_TILE; ++t) {
                FETCH_FLOAT4(a_reg[write_idx][4 * t]) =
                        FETCH_FLOAT4(As[write_idx][i][ty / THREAD_TILE + t * BM / THREAD_TILE]);
            }
#pragma unroll
            for (int t = 0; t < THREAD_TILE; ++t) {
                FETCH_FLOAT4(b_reg[write_idx][t * 4]) =
                        FETCH_FLOAT4(Bs[write_idx][i][tx / THREAD_TILE + t * BM / THREAD_TILE]);
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += a_reg[write_idx][m] * b_reg[write_idx][n];
                }
            }
        }
        write_idx ^= 1;
    }

#pragma unroll
    for (int m = 0; m < TM / 2; ++m) {
        FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(ty / 2 + m, tx / 2, N)]);
        load_a_cache[0] = alpha * accum[m][0] + beta * load_a_cache[0];
        load_a_cache[1] = alpha * accum[m][1] + beta * load_a_cache[1];
        load_a_cache[2] = alpha * accum[m][2] + beta * load_a_cache[2];
        load_a_cache[3] = alpha * accum[m][3] + beta * load_a_cache[3];
        FETCH_FLOAT4(C[OFFSET(ty / 2 + m, tx / 2, N)]) = FETCH_FLOAT4(load_a_cache);
        FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(ty / 2 + m, tx / 2, N) + BN / 2]);
        load_a_cache[0] = alpha * accum[m][4] + beta * load_a_cache[0];
        load_a_cache[1] = alpha * accum[m][5] + beta * load_a_cache[1];
        load_a_cache[2] = alpha * accum[m][6] + beta * load_a_cache[2];
        load_a_cache[3] = alpha * accum[m][7] + beta * load_a_cache[3];
        FETCH_FLOAT4(C[OFFSET(ty / 2 + m, tx / 2, N) + BN / 2]) = FETCH_FLOAT4(load_a_cache);
    }

#pragma unroll
    for (int m = 0; m < TM / 2; ++m) {
        FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(ty / 2 + m + BN / 2, tx / 2, N)]);
        load_a_cache[0] = alpha * accum[m + TM / 2][0] + beta * load_a_cache[0];
        load_a_cache[1] = alpha * accum[m + TM / 2][1] + beta * load_a_cache[1];
        load_a_cache[2] = alpha * accum[m + TM / 2][2] + beta * load_a_cache[2];
        load_a_cache[3] = alpha * accum[m + TM / 2][3] + beta * load_a_cache[3];
        FETCH_FLOAT4(C[OFFSET(ty / 2 + m + BN / 2, tx / 2, N)]) = FETCH_FLOAT4(load_a_cache);
        FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(ty / 2 + m + BN / 2, tx / 2, N) + BN / 2]);
        load_a_cache[0] = alpha * accum[m + TM / 2][4] + beta * load_a_cache[0];
        load_a_cache[1] = alpha * accum[m + TM / 2][5] + beta * load_a_cache[1];
        load_a_cache[2] = alpha * accum[m + TM / 2][6] + beta * load_a_cache[2];
        load_a_cache[3] = alpha * accum[m + TM / 2][7] + beta * load_a_cache[3];
        FETCH_FLOAT4(C[OFFSET(ty / 2 + m + BN / 2, tx / 2, N) + BN / 2]) = FETCH_FLOAT4(load_a_cache);
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
    no_share_conflict_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
