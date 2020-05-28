#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stddef.h>

extern "C"

void conv2D(int PW, int PH, int KW, int KH, int IC, int OC, int SW, int SH, int OW, int OH, float *I, float *W, float *O)
{
	// cuBLAS implementation of 2D convolution
	// cuBLAS's Sgemm is executed based on that the matrix are ordered in Column Major.
	// Store matA and matB in column major order and caculate matrix multiplication and store the result to matC.
	// After done convolution, store matC to Output matrix in row major order.

	// Since unexpected calculation error occurs when use transpose approach to handle RowMajor <-> ColumnMajor,
	// Just reuse openBLAS code that properly working but different major.
	
	int o_size = OW * OH, k_size = KW * KH;
	int r_idx, c_idx, b_idx, w_idx;
	float *matA, *matB, *matC;

	cublasHandle_t handle;
	float al = 1.0f, bet = 1.0f;
	float *d_a, *d_b, *d_o;
	
	matC = (float *)malloc((o_size * OC) * sizeof(float));
	for(int i = 0; i < o_size * OC; i++)
		matC[i] = 0;	


	for(int ic = 0; ic < IC; ic++)
	{
		matA = (float *)malloc((o_size * k_size) * sizeof(float));

		for(int ow = 0; ow < OW; ow++)
		{
			for(int oh  = 0; oh < OH; oh++)
			{
				r_idx = ow * OH + oh;
				for(int i = 0; i < KW; i++)
				{
					for(int j = 0; j < KH; j++)
					{
						c_idx = i * KH + j;
						matA[c_idx * o_size + r_idx] = I[((ow * SW * PH + oh * SH) + (i * PH + j)) * IC + ic];
					}
				}
			}
		}

		matB = (float *)malloc((k_size * OC) * sizeof(float));
		
		for(int i = 0; i < KW; i++)
		{
			for(int j = 0; j < KH; j++)
			{		
				b_idx = i * KH + j;
				for(int oc = 0; oc < OC; oc++)
				{
					w_idx = (b_idx * (IC * OC)) + ic * OC + oc;
					matB[oc * k_size + b_idx] = W[w_idx];
				}
			}
		}

		cudaMalloc(&d_a, (o_size * k_size) * sizeof(float));
		cudaMalloc(&d_b, (k_size * OC) * sizeof(float));
		cudaMalloc(&d_o, (o_size * OC) * sizeof(float));

		cublasSetMatrix(o_size, k_size, sizeof(float), matA, o_size, d_a, o_size);
		cublasSetMatrix(k_size, OC,  sizeof(float), matB, k_size, d_b, k_size);
		cublasSetMatrix(o_size, OC, sizeof(float), matC, o_size, d_o, o_size);

		cublasCreate(&handle);

		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				o_size, OC, k_size,
				&al,
				d_a, o_size,
				d_b, k_size,
				&bet,
				d_o, o_size);
		
		cublasGetMatrix(o_size, OC, sizeof(float), d_o, o_size, matC, o_size);
		
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_o);
		free(matA);
		free(matB);
	}
	
	for(int i = 0; i < o_size; i++)
		for(int j = 0; j < OC; j++)
			O[i * OC + j] = matC[j * o_size + i];

	free(matC);
	cublasDestroy(handle);
}