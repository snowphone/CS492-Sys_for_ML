#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cblas.h>
#include <math.h>

extern "C"

void conv2D(int PW, int PH, int KW, int KH, int IC, int OC, int SW, int SH, int OW, int OH, float *I, float *W, float *O)
{
	// cuBLAS implementation of 2D convolution
	// cuBLAS's Sgemm is executed based on that the matrix are ordered in Column Major.
	// Since tiled_in, weight and output are stored in RowMajor order, their transpose is ColMajor order.
	// Therefore, for the RowMajor order, we transposed the equation A X B = C  ->  B^T X A^T = C^T.
	// Then the only change from openBLAS are 1. switching tiled_in and weight 2. Memory copy between host and GPU
	
	int o_size = OW * OH, k_size = KW * KH;
	int r_idx, c_idx;
	float *tiled_in;

	cublasHandle_t handle;
	float al = 1.0f, bet = 1.0f;
	float *d_tiled_in, *d_w, *d_o;
	
	cudaMalloc(&d_tiled_in, (k_size * IC * o_size) * sizeof(float));
	cudaMalloc(&d_w, (OC * IC * k_size) * sizeof(float));
	cudaMalloc(&d_o, (OC * o_size) * sizeof(float));

	tiled_in = (float *)malloc((k_size * IC * o_size) * sizeof(float));

	for(int ow = 0; ow < OW; ow++)
		for(int oh  = 0; oh < OH; oh++)
		{
			r_idx = ow * OH + oh;
			for(int ic = 0; ic < IC ; ic++)
				for(int i = 0; i < KW; i++)
					for(int j = 0; j < KH; j++)
					{	
						c_idx = i * KH + j;
						tiled_in[(r_idx * IC + ic) * k_size + c_idx] = I[((ow * SW * PH + oh * SH) + (i * PH + j)) * IC + ic];
					}
		}
		
	cublasSetMatrix(k_size * IC, o_size, sizeof(float), tiled_in, k_size * IC, d_tiled_in, k_size * IC);
	cublasSetMatrix(OC * IC, k_size,  sizeof(float), W, OC * IC, d_w, OC * IC);
	cublasSetMatrix(OC, o_size, sizeof(float), O, OC, d_o, OC);

	cublasCreate(&handle);
	for(int i = 0; i < IC; i++)
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				OC, o_size, k_size,
				&al,
				d_w + i * OC, OC * IC,
				d_tiled_in + i * k_size, k_size * IC,
				&bet,
				d_o, OC);
	
	cublasGetMatrix(OC, o_size, sizeof(float), d_o, OC, O, OC);
	
	cudaFree(d_tiled_in);
	cudaFree(d_w);
	cudaFree(d_o);
	cublasDestroy(handle);
	free(tiled_in);
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

