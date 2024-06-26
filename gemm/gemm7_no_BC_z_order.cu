#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <mma.h>


constexpr size_t BLOCK_SIZE = 16; // we assume that every block has equal blockDim.x and blockDim.y
constexpr size_t BLOCK_M = 128;   // These const values decide how many thing a thread compute and the amount of shared memory to allocate.
constexpr size_t BLOCK_N = 128;
constexpr size_t BLOCK_K = 8; // don't set 64 here, it will cause bank conflict and lower occupancy.
constexpr size_t BLOCK_M_COMPUTE = BLOCK_M / BLOCK_SIZE;
constexpr size_t BLOCK_N_COMPUTE = BLOCK_N / BLOCK_SIZE;

#define colM(a, i, j, lda) a[((j) * (lda)) + (i)]
#define rowM(a, i, j, lda) a[(j) + (i) * (lda)]

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
    const size_t baseX = blockIdx.x * blockDim.x * BLOCK_M_COMPUTE;
    const size_t baseY = blockIdx.y * blockDim.y * BLOCK_N_COMPUTE;

    const size_t baseIdx = threadIdx.y * blockDim.x + threadIdx.x;

    float c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};
    constexpr size_t subAlda = BLOCK_M + 4; // plus 4 here to avoid bank conflict and maintain float4 read

    __shared__ float subA[subAlda * BLOCK_K];
    __shared__ float subB[BLOCK_N * BLOCK_K];

    float4 regB[BLOCK_M_COMPUTE / 4]; // hopefully, these should reside in register.
    float4 regA[BLOCK_M_COMPUTE / 4];

    const float *baseA = A + baseY * K;
    const float *baseB = B + baseX;

    // 利用位运算 计算很巧妙！
    // 从A中读取到As中 将Bk分为4-4 按照 thread_id 优先完成BK维度 再完成BM维度
    // rowB 需要 除以32是因为 BN=128 / 4 = 32
    // 0...,001,111,111
    // x,xxx,xxx -> x,xxx,x00 只需要最后七位是因为BN为128(0-127)，最多就是127，<< 可能会导致有符号数的符号位受到影响
    int rowA = baseIdx >> 1, rowB = baseIdx >> 5, colA = (baseIdx & 1) << 2, colB = (baseIdx << 2) & 127;
    int warpId = baseIdx >> 5, warpBaseId = baseIdx & 31;
    int rowC = ((warpId >> 1 << 3) + ((warpBaseId >> 4) << 1) + (warpBaseId & 1)) << 2, colC = (((warpId & 1) << 4) + ((warpBaseId & 15) >> 1)) << 2;
    float *baseC = C + (baseY + rowC) * N + baseX + colC;

    for (int i = 0; i < K; i += BLOCK_K)
    {
        regB[0] = *reinterpret_cast<const float4 *>(baseB + i * N + rowB * N + colB);
        regA[0] = *reinterpret_cast<const float4 *>(baseA + i + rowA * K + colA);
        *reinterpret_cast<float4 *>(&subB[baseIdx * 4]) = regB[0];
        subA[rowA + colA * subAlda] = regA[0].x;
        subA[rowA + (colA + 1) * subAlda] = regA[0].y;
        subA[rowA + (colA + 2) * subAlda] = regA[0].z;
        subA[rowA + (colA + 3) * subAlda] = regA[0].w;

        __syncthreads();
#pragma unroll
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
            regB[0] = *reinterpret_cast<float4 *>(&subB[colC + BLOCK_N * ii]);
            regB[1] = *reinterpret_cast<float4 *>(&subB[colC + 32 + BLOCK_N * ii]);

            regA[0] = *reinterpret_cast<float4 *>(&subA[rowC + ii * subAlda]);
            regA[1] = *reinterpret_cast<float4 *>(&subA[(rowC + 16) + ii * subAlda]);

#pragma unroll
            for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
            {
#pragma unroll
                for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
                {
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].x * regB[cpj].x;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].y * regB[cpj].x;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].z * regB[cpj].x;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].w * regB[cpj].x;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * N]);
        regA[0].x = regA[0].x * beta + alpha * c[i * BLOCK_N_COMPUTE];
        regA[0].y = regA[0].y * beta + alpha * c[1 + i * BLOCK_N_COMPUTE];
        regA[0].z = regA[0].z * beta + alpha * c[2 + i * BLOCK_N_COMPUTE];
        regA[0].w = regA[0].w * beta + alpha * c[3 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[i * N]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * N + 32]);
        regA[0].x = regA[0].x * beta + alpha * c[4 + i * BLOCK_N_COMPUTE];
        regA[0].y = regA[0].y * beta + alpha * c[5 + i * BLOCK_N_COMPUTE];
        regA[0].z = regA[0].z * beta + alpha * c[6 + i * BLOCK_N_COMPUTE];
        regA[0].w = regA[0].w * beta + alpha * c[7 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[i * N + 32]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * N]);
        regA[0].x = regA[0].x * beta + alpha * c[32 + i * BLOCK_N_COMPUTE];
        regA[0].y = regA[0].y * beta + alpha * c[33 + i * BLOCK_N_COMPUTE];
        regA[0].z = regA[0].z * beta + alpha * c[34 + i * BLOCK_N_COMPUTE];
        regA[0].w = regA[0].w * beta + alpha * c[35 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * N]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * N + 32]);
        regA[0].x = regA[0].x * beta + alpha * c[36 + i * BLOCK_N_COMPUTE];
        regA[0].y = regA[0].y * beta + alpha * c[37 + i * BLOCK_N_COMPUTE];
        regA[0].z = regA[0].z * beta + alpha * c[38 + i * BLOCK_N_COMPUTE];
        regA[0].w = regA[0].w * beta + alpha * c[39 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * N + 32]) = *reinterpret_cast<float4 *>(&regA[0]);
    }
}

void MY_MMult(cublasHandle_t handle, int M, int N, int K, float *A, int lda,
              float *B, int ldb, float *C, int ldc) {

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
 
    const int alpha = 1.0;
    const int beta = 0.0;

#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
#endif
}

