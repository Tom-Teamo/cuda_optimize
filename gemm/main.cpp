#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>
#include <cublas_v2.h>

#include "parameters.h"
#include "helper.h"

void REF_MMult(int, int, int, float *, int, float *, int, float *, int);
void MY_MMult(cublasHandle_t, int, int, int, float *, int, float *, int,
              float *, int);

int main() {

    // print gpu info
    cudaDeviceProp device_prop;
    int device_id = 0;
    checkCudaError(cudaSetDevice(device_id));
    checkCudaError(cudaGetDeviceProperties(&device_prop, device_id));

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", 
            device_id, device_prop.name, device_prop.major, device_prop.minor);

    int p, m, n, k, rep;
    double dtime, dtime_best, gflops, diff;
    float *a, *b, *cref, *cold;

    printf("MY_MMult = [\n");

    // cublas must be initialized before any cublas library called
    cublasHandle_t handle;
    checkCudaError(cublasCreate(&handle));

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    for (p = PFIRST; p <= PLAST; p += PINC) {
        m = (M == -1? p: M);
        n = (N == -1? p: N);
        k = (K == -1? p: K);

        gflops = 2.0 * m * n * k * 1.0e-09;

        const int lda = k, ldb = n, ldc = n;

        const size_t size_A = m * k * sizeof(float);
        const size_t size_B = k * n * sizeof(float);
        const size_t size_C = m * n * sizeof(float);
        a = (float *)malloc(size_C);
        b = (float *)malloc(size_B);
        cold = (float *)malloc(size_C);
        cref = (float *)malloc(size_C);

        // generate random matrix
        random_matrix(m, k, a, m);
        random_matrix(k, n, b, k);
        memset(cold, 0, size_C);
        memset(cref, 0, size_C);

        // init device matrix
        float *d_A, *d_B, *d_C, *d_c_ref;
        checkCudaError(cudaMalloc((void **)&d_A, size_A));
        checkCudaError(cudaMalloc((void **)&d_B, size_B));
        checkCudaError(cudaMemcpy(d_A, a, size_A, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(d_B, b, size_B, cudaMemcpyHostToDevice));
        checkCudaError(cudaMalloc((void **)&d_C, size_C));
        checkCudaError(cudaMalloc((void **)&d_c_ref, size_C));

        // // run the reference implementation
        // REF_MMult(m, n, k, a, lda, b, ldb, cref, ldc);

        // time start
        checkCudaError(cudaEventRecord(start, NULL));

        // run the my implementation
        for (rep = 0; rep < NREPEATS; rep++) {
            MY_MMult(handle, m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
        }

        // time stop
        checkCudaError(cudaEventRecord(stop, NULL));
        // wait for the event to complete
        checkCudaError(cudaEventSynchronize(stop));
        float msec_total = 0.0;
        checkCudaError(cudaEventElapsedTime(&msec_total, start, stop));

        // compute and print the performance
        float msec_per_mmtul = msec_total / NREPEATS;
        // A * B 一共需要m * n * k次乘加运算
        double flops_per_mmul = 2.0 * m * n * k;
        double gflops = (flops_per_mmul * 1.0e-9f) / (msec_per_mmtul / 1000.0f);

        // copy result from device to host
        checkCudaError(cudaMemcpy(cold, d_C, size_C, cudaMemcpyDeviceToHost));

        // cuBlas
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCudaError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                    &alpha, d_B, n, d_A, k, &beta, d_c_ref, n));
        checkCudaError(cudaMemcpy(cref, d_c_ref, size_C, cudaMemcpyDeviceToHost));
        
        // compare the results
        diff = compare_metrices(m, n, cold, ldc, cref, ldc);

        printf("%d %.2f %le \n", p, gflops, diff);

        free(a);
        free(b);
        free(cold);
        free(cref);
        
        checkCudaError(cudaFree(d_A));
        checkCudaError(cudaFree(d_B));
        checkCudaError(cudaFree(d_C));
    }

    checkCudaError(cublasDestroy(handle));

    printf("];\n");
    return 0;
}



