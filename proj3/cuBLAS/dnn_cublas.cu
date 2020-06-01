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
	// Store tiled_in in column major order and caculate matrix multiplication and store the result to O_colMaj.
	// After done convolution, store O_colMaj to Output matrix in row major order.

	// Since unexpected calculation error occurs when use transpose approach to handle RowMajor <-> ColumnMajor,
	// Just reuse openBLAS code that properly working but different major.
	
	int o_size = OW * OH, k_size = KW * KH;
	int r_idx, c_idx;
	float *tiled_in, *O_colMaj;

	cublasHandle_t handle;
	float al = 1.0f, bet = 1.0f;
	float *d_a, *d_b, *d_o;
	
	cudaMalloc(&d_a, (o_size * k_size * IC) * sizeof(float));
	cudaMalloc(&d_b, (OC * IC * k_size) * sizeof(float));
	cudaMalloc(&d_o, (o_size * OC) * sizeof(float));

	tiled_in = (float *)malloc((o_size * k_size * IC) * sizeof(float));
	O_colMaj = (float *)malloc((o_size * OC * IC) * sizeof(float));

	for(int i = 0; i < o_size * OC; i++)
		O_colMaj[i] = 0;	
	
	for(int ow = 0; ow < OW; ow++)
		for(int oh  = 0; oh < OH; oh++)
		{
			r_idx = ow * OH + oh;
			for(int ic = 0; ic < IC ; ic++)
				for(int i = 0; i < KW; i++)
					for(int j = 0; j < KH; j++)
					{	
						c_idx = i * KH + j;
						tiled_in[(c_idx * IC + ic) * o_size + r_idx] = I[((ow * SW * PH + oh * SH) + (i * PH + j)) * IC + ic];
					}
		}
		
	cublasSetMatrix(o_size * IC, k_size, sizeof(float), tiled_in, o_size * IC, d_a, o_size * IC);
	cublasSetMatrix(OC * IC, k_size,  sizeof(float), W, OC * IC, d_b, OC * IC);
	cublasSetMatrix(o_size, OC, sizeof(float), O_colMaj, o_size, d_o, o_size);

	cublasCreate(&handle);
	for(int i = 0; i < IC; i++)
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
				o_size, OC, k_size,
				&al,
				d_a + i * o_size, o_size * IC,
				d_b + i * OC, OC * IC,
				&bet,
				d_o, o_size);
		
	cublasGetMatrix(o_size, OC, sizeof(float), d_o, o_size, O_colMaj, o_size);
	
	for(int i = 0; i < o_size; i++)
		for(int j = 0; j < OC; j++)
			O[i * OC + j] = O_colMaj[j * o_size + i];
		
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_o);
	cublasDestroy(handle);
	free(tiled_in);
	free(O_colMaj);
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

void maxPool2D(int PW, int PH, int KW, int KH, int OC, int SW, int SH, double *I, double *O)
{
        double max;
        int o_idx, s_idx, k_idx;
        int OW = PW/SW, OH = PH/SH;

        for(int oc = 0; oc < OC; oc++)
                for(int ow = 0; ow < OW; ow++)
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

