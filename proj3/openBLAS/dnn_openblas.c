#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>

void conv2D(int PW, int PH, int KW, int KH, int IC, int OC, int SW, int SH, int OW, int OH, double *I, double *W, double *O)
{
	// openBLAS implementation of 2D convolution
	// Simillar approach with Toepliz Matrix
	// matA is [output size X kernel size] matrix. Each row is corresponding input values that multiplied with the kernel to produce one output.
	// matB is [kernel size X output channel] matrix. Flatten each kenels to a column and concatenate all of them. (same as output channels)
	// By multiplying matA and matB, we can produce output values of each input channel.
	// Iterating as the number of input channels and accumulate all of the results, the convolution is done.
	// In the matrix multiplication, we set beta value as 1, so the dgemm does C = A x B + C, so accumulation is naturally done during matrix multiplication.

	int o_size = OW * OH, k_size = KW * KH;
	int r_idx, c_idx, b_idx, w_idx;

	for(int ic = 0; ic < IC; ic++)
	{
		double* matA = malloc((o_size * k_size) * sizeof(double));

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
						matA[r_idx * k_size + c_idx] = I[((ow * SW * PH + oh * SH) + (i * PH + j)) * IC + ic];
					}
				}
			}
		}

		double* matB = malloc((k_size * OC) * sizeof(double));
		
		for(int i = 0; i < KW; i++)
		{
			for(int j = 0; j < KH; j++)
			{		
				b_idx = i * KH + j;
				for(int oc = 0; oc < OC; oc++)
				{
					w_idx = (b_idx * (IC * OC)) + ic * OC + oc;
					matB[b_idx * OC + oc] = W[w_idx];
				}
			}
		}

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				o_size, OC, k_size,
				1,
				matA, k_size,
				matB, OC,
				1,
				O, OC
			   );
			
		free(matA);
		free(matB);
	}
}

void biasAdd(int size, int OC, double *I, double *B, double *O)
{
	for(int i = 0; i < size; i++)
		for(int oc = 0; oc < OC; oc++)
		{
			O[i * OC + oc] = I[i * OC + oc] + B[oc];
		}
}

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

void batchNorm(int size, int OC, double *I, double *mean, double *gamma, double *variance, double epsilon, double *O)
{	
	for(int i = 0; i < size; i++)
		for(int oc = 0; oc < OC; oc++)
		{	
			O[i * OC + oc] = (I[i * OC + oc] - mean[oc]) * gamma[oc] / sqrt(variance[oc] = epsilon);	
		}
}

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
