#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stddef.h>
#include <cblas.h>
#include <math.h>

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
	
	cudaMalloc(&d_a, (o_size * k_size) * sizeof(float));
	cudaMalloc(&d_b, (k_size * OC) * sizeof(float));
	cudaMalloc(&d_o, (o_size * OC) * sizeof(float));

	matA = (float *)malloc((o_size * k_size) * sizeof(float));
	matB = (float *)malloc((k_size * OC) * sizeof(float));
	matC = (float *)malloc((o_size * OC) * sizeof(float));

	for(int i = 0; i < o_size * OC; i++)
		matC[i] = 0;	

	for(int ic = 0; ic < IC; ic++)
	{
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
		}
	
	for(int i = 0; i < o_size; i++)
		for(int j = 0; j < OC; j++)
			O[i * OC + j] = matC[j * o_size + i];
		
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_o);
	cublasDestroy(handle);
	free(matA);
	free(matB);
	free(matC);
}

extern "C"

void biasAdd(int size, int OC, double *I, double *B, double *O)
{
	for(int i = 0; i < size; i++)
	{
		cblas_daxpy(OC, 1, B, 1, O + i * OC, 1);
		cblas_daxpy(OC, 1, I + i * OC, 1, O + i * OC, 1);
	}
}

extern "C"

void maxPool2D(int PW, int PH, int KW, int KH, int OC, int SW, int SH, int OW, int OH, double *I, double *O)
{
        double max;
        int o_idx, s_idx, k_idx;

        for(int oc = 0; oc < OC; oc++)
        {
                for(int ow = 0; ow < OW; ow++)
                {
                        for(int oh  = 0; oh < OH; oh++)
                        {
                                o_idx = (ow * OH + oh) * OC + oc;
                                s_idx = (ow * SW * PH + oh * SH) * OC + oc;
                                max = I[s_idx];
                                for(int i = 0; i < KW; i++)
                                {
                                        if(ow * SW + i >= PW)
                                                break;
                                        for(int j = 0; j < KH; j++)
                                        {
                                                if(oh * SH + j >= PH)
                                                        break;
                                                k_idx = (i * PH + j) * OC;
                                                if(I[s_idx + k_idx] > max)
                                                        max = I[s_idx + k_idx];
                                        }
                                }
                                O[o_idx] = max;
                        }
                }
        }
}

extern "C"

void batchNorm(int size, int OC, double *I, double *mean, double *gamma, double *variance, double epsilon, double *O)
{
	double coeff;

	for(int i = 0; i < size; i++)
	{
		cblas_daxpy(OC, -1, mean, 1, O + i * OC, 1);
		cblas_daxpy(OC, 1, I + i * OC, 1, O + i * OC, 1);
	}
	
	for(int oc = 0; oc < OC; oc++)
	{
		coeff = gamma[oc] / sqrt(variance[oc] + epsilon);
	        cblas_dscal(size, coeff, O + oc, OC);
	}
}

extern "C"

void leakyReLU(int size, int OC, double *I, double *O)
{
        int idx;

        for(int i = 0; i < size; i++)
                for(int oc = 0; oc < OC; oc++)
                {
                        idx = i * OC + oc;
                        if(I[idx] < 0)
                                I[idx] = I[idx] * 0.1;
                        O[idx] = I[idx];
                }
}

