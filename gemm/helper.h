#ifndef HELPER_H
#define HELPER_H

#include <stdlib.h>
#include <stdio.h>


template <typename T>
void check(T result, char const *const func, 
           const char *const file, int const line)
{
    if (result) {
        printf("CUDA error at %s:%d code=%d \"%s\" \n", 
                file, line, static_cast<unsigned int>(result), func);
    }
}

#define checkCudaError(val) check(val, #val, __FILE__, __LINE__);


void random_matrix(int m, int n, float *a, int lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i + j * lda] = 2 * (float)drand48() - 1.0;
        }
    }
}

/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* 
    Routine for computing C = A * B
    (m,k)*(k,n)=(m,n)
*/
void REF_MMult( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
    int i, j, p;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            for (p = 0; p < k; p++)
                C(i,j) = C(i, j) + A(i,p) * B(p,j);
    
}


#define my_abs( x ) ( (x) < 0.0 ? -(x) : (x) )

float compare_metrices(int m, int n, float *a, int lda, 
                                     float *b, int ldb) {
    int i, j;
    float max_diff = 0.0, diff;
    bool print = true;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            diff = my_abs(A(i,j) - B(i,j));
            max_diff = (diff > max_diff)? diff: max_diff;
            if (print && (max_diff > 0.5 || max_diff < -0.5)) {
                printf("\n error: i %d  j %d diff %f", i, j, max_diff);
                print = false;
            }
        }
    }

    return max_diff;
}

#endif