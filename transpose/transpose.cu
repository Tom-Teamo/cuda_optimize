#include <cuda.h>
#include <cstdlib>
#include <cublas.h>
#include <iostream>

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void transpose_v0(int M, int N,
                             const float *__restrict__ iA,
                             float *__restrict__ oA) {
    const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
  
    if (x >= N || y >= M) {
        return;
    }

    oA[x * M + y] = iA[y * N + x];
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void transpose_v1(int M, int N,
                             const float *__restrict__ iA,
                             float *__restrict__ oA) {
    __shared__ float s_mem[32][32];
    
    const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    
    if (x >= N || y >= M) {
        return;
    }

    float target = iA[y * N + x];
    
    // transpose
    s_mem[threadIdx.x][threadIdx.y] = target;

    __syncthreads();

    float* o_start = &oA[blockIdx.x * BLOCK_DIM_X * M + blockIdx.y * BLOCK_DIM_Y];

    o_start[threadIdx.y * M + threadIdx.x] = s_mem[threadIdx.y][threadIdx.x];
}


__host__ void launch_kernel(int which_kernel, const float* iA, float* oA, int M, int N) {
    switch (which_kernel) {

        case 0:{
            const int block_x = 32, block_y = 32;
            dim3 grid_dim((M + block_x - 1) / block_x, (N + block_y - 1) / block_y);
            dim3 block_dim(block_x, block_y);

            transpose_v0<block_x, block_y><<<grid_dim, block_dim>>>(M, N, iA, oA);
            break;
        }
        case 1: {
            const int block_x = 32, block_y = 8;
            dim3 grid_dim((M + block_x - 1) / block_x, (N + block_y - 1) / block_y);
            dim3 block_dim(block_x, block_y);

            transpose_v1<block_x, block_y><<<grid_dim, block_dim>>>(M, N, iA, oA);
            break;
        }
        default:
            break;
    }
}

__host__ bool check_result(float* iA, float* oA, int M, int N) {
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            if (iA[i * N + j] != oA[j * M + i]) {
                return false;
            }
        }
    }

    return true;
}

int main() {
    const int which_kerner = 1;
    const int M = 16384, N = 16384;
    // const int M = 1024, N = 1024;
    const int matrix_size = sizeof(float) * M * N;
    const int matrix_element_num = M * N;

    float* host_iA = (float*)malloc(matrix_size);
    float* host_oA = (float*)malloc(matrix_size);

    for (size_t i = 0; i < matrix_element_num; i++)
    {
        host_iA[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    float* device_iA;
    float* device_oA;
    cudaMalloc((void**)&device_iA, matrix_size);
    cudaMalloc((void**)&device_oA, matrix_size);

    cudaMemcpy(device_iA, host_iA, matrix_size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    
    launch_kernel(which_kerner, device_iA, device_oA, M , N);
    cudaMemcpy(host_oA, device_oA, matrix_size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if (check_result(host_iA, host_oA, M, N)) {
        std::cout << "----------------Test Pass----------------" << std::endl;
    }
    else {
        std::cout << "----------------Test Fail----------------" << std::endl;
    }
    
    free(host_iA);
    free(host_oA);
    cudaFree(device_iA);
    cudaFree(device_oA);

    return 0;
}