#include <stdio.h>
#include <cblas.h>

void mat_mul(int M, int N, int K, double *A, double *B, double *C)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			M, N, K, 
			1,
			A, K,
			B, N,
			0,
			C, M
		   );
}

