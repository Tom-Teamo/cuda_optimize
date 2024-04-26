#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
cublas<t>gemm()
    cublasSgmm  ---->   float
    cublasDgmm  ---->   double
    cublasCgmm  ---->   cuComplex
    cublasZgmm  ---->   cuDoubleComplex
    cublasHgmm  ---->   __half

*/

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              d_B, n, d_A, k, &beta, d_C, n);
}
