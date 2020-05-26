#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>

void conv2D(int PW, int PH, int KW, int KH, int IC, int OC, int SW, int SH, int OW, int OH, double *I, double *W, double *O)
{
	// openBLAS implementation of 2D convolution
	// Simillar approach with Toepliz Matrix
	// matA is [output size X kernel size] matrix. Each row is corresponding input values that multiplied with the kernel to produce one output.
	// matB is [kernel size X output channel] matrix. Flatten kenel to a column and concatenate all output channels.
	// By multiplying matA and matB, we can produce output values of each input channel.
	// Iterating the input channel and accumulate all of the results, the convolution is done.
	// In the matrix multiplication, we set beta value as 1, so the dgemm does C = A x B + C, so accumulation is naturally done.

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
