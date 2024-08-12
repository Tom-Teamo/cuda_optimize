#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>

/*
    naive implementation

    fused multiply add instruction: 需要启用的话 开启 --fmad=true
        > https://stackoverflow.com/questions/58989033/how-to-use-fma-on-gpu-v100
*/
template <int block_size>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
    int x = block_size * blockIdx.x + threadIdx.x;
    int y = block_size * blockIdx.y + threadIdx.y;

    // if (y < m && x < n) {
    //     float sum = 0.0;
    //     for (int i = 0; i < k; i++) {
    //         sum += a[y * lda + i] * b[i * ldb + x];
    //     }
    //     c[y * ldc + x] = sum;
    // }

    /*
        if we exchange x and y, we will get a bad performance
        because that c[], x changes fast than y, so c[] is not 连续
    */
    if (y < m && x < n) {
        float sum = 0.0;
        for (int i = 0; i < k; i++) {
            sum += a[x * lda + i] * b[i * ldb + x];
        }
        c[y * ldc + y] = sum;
    }

}


void MY_MMult(cublasHandle_t handle, int m, int n, int k,
            float *A, int lda, float *B, int ldb, float *C, int ldc) {
    constexpr int block_size = 16;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid((m + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    sgemm<block_size><<<dim_grid, dim_block>>>(m, n, k, A, lda, B, ldb, C, ldc);
}