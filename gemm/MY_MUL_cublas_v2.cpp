#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
This function is an extension of cublas<t>gemm that 
    allows the user to individually specify the data types for each of the A, B and C matrices,
    the precision of computation and the GEMM algorithm to be run.
*/

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  const float alpha = 1.0f;
  const float beta = 0.0f;
#if __CUDACC_VER_MAJOR__ >= 11
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#else
    cudaDataType_t compute_type = CUDA_R_32F;
#endif

cublasGemmEx(
    handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
    (void*)(&alpha), d_B, CUDA_R_32F, n, d_A, CUDA_R_32F, k,
    (void*)(&beta), d_C, CUDA_R_32F, n, compute_type, CUBLAS_GEMM_DEFAULT);
}
