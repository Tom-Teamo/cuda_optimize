#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <mma.h>

#include <cublas_v2.h>

using namespace nvcuda;

#define uint unsigned int
#define coreSizeM 16
#define coreSizeN 16
#define coreSizeK 16

/*
    half: FP16
*/

__global__ void TensorCoreMM(half* a, half* b, float* c, const int M, const int N, const int K) {

    const uint warp_M = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    const uint warp_N = blockDim.y * blockIdx.y + threadIdx.y;

    const uint la = K, lb = N, lc = N;

    const uint aRow = warp_M * coreSizeM; // 当前tile左上角在A上的行数
    const uint bCol = warp_N * coreSizeN; // 当前tile左上角在B上的列数

    if (aRow >= M || bCol >= N) return;

    // 声明 fragment
    wmma::fragment<wmma::matrix_a, coreSizeM, coreSizeN, coreSizeK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, coreSizeM, coreSizeN, coreSizeK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, coreSizeM, coreSizeN, coreSizeK, float> c_frag;

    // 清理 c_frag
    wmma::fill_fragment(c_frag, 0.f);

    for (int i = 0; i < la; i += coreSizeK) {

        size_t aCol = i;
        size_t bRow = i;

        // load
        if (aCol < K && bRow < K) {
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * la, la);
            wmma::load_matrix_sync(b_frag, b + bCol + bRow * lb, lb);
            // multiple and accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

 
    if (aRow < M && bCol < N) {
        // wmma::store_matrix_sync(c + cCol + cRow * N, c_frag, N, wmma::mem_row_major);
        wmma::store_matrix_sync(c + bCol + aRow * lc, c_frag, lc, wmma::mem_row_major);
    }

    // store
    // wmma::store_matrix_sync(c + bCol + aRow * lc, c_frag, lc, wmma::mem_row_major);
    
}


#define vectorPtr(x) thrust::raw_pointer_cast(x.data())

int main() {

    // C = A * B + C
    const float alpha = 1;
    const float beta = 0;

    size_t M = 1024, N = 1024, K = 1024;

    thrust::host_vector<float> A_host(M * K);
    thrust::host_vector<float> B_host(K * N);
    thrust::host_vector<float> C_host_TC(M * N, 0.f);

    // Thrust’s sequence function can be used to create a sequence of equally spaced values
    // 0, 1, 2, 3, 4 ...
    // thrust::sequence(A_float.begin(), A_float.end());
    // thrust::fill(A_host.begin(), A_host.end(), 2 * (float)drand48() - 1.0);
    // thrust::fill(B_host.begin(), B_host.end(), 2 * (float)drand48() - 1.0);
    thrust::fill(A_host.begin(), A_host.end(), 2);
    thrust::fill(B_host.begin(), B_host.end(), 1);

    thrust::device_vector<half> A_device_half(A_host.begin(), A_host.end());
    thrust::device_vector<half> B_device_half(B_host.begin(), B_host.end());
    thrust::device_vector<float> C_device(M * N, 0.f);

    dim3 blockSize(128, 4);
    // 8 256
    dim3 gridSize(2 * (M + blockSize.x - 1) / blockSize.x, 16);

    TensorCoreMM<<<gridSize, blockSize>>>(vectorPtr(A_device_half), vectorPtr(B_device_half), vectorPtr(C_device), M, N, K);


    for (int i = 0; i < M; ++i) {
        // thrust::copy(C.begin() + i * N, C.begin() + (i + 1) * N, std::ostream_iterator<float>(std::cout, ", "));
        thrust::copy(C_device.begin() + i * N, C_device.begin() + (i + 1) * N, C_host_TC.begin() + i * N);
    }
    
    // calculate using cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    thrust::device_vector<float> B_device_float(B_host.begin(), B_host.end());
    thrust::device_vector<float> A_device_float(A_host.begin(), A_host.end());
    thrust::device_vector<float> C_device_cublas(M * N, 0.f);
    thrust::host_vector<float> C_host_cublas(M * N, 1.f);
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              vectorPtr(B_device_float), N, vectorPtr(A_device_float), K, &beta, vectorPtr(C_device_cublas), N);
    
    for (size_t i = 0; i < M; ++i) {
        thrust::copy(C_device_cublas.begin() + i * N, C_device_cublas.begin() + (i + 1) * N, C_host_cublas.begin() + i * N);
    }
    
    for (size_t i = 0; i < M * N; i++)
    {
        if (abs(C_host_cublas[i] - C_host_TC[i]) > 1) {
            std::cout << "error at (" << i << ") " << " cublas:" << C_host_cublas[i] << " TC:" << C_host_TC[i] << std::endl;
            break;
        }
    }
    
    return 0;

}