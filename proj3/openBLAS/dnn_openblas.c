#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <math.h>

void conv2D(int PW, int PH, int KW, int KH, int IC, int OC, int SW, int SH, int OW, int OH, double *I, double *W, double *O)
{
	// openBLAS implementation of 2D convolution
	// Simillar approach with Toepliz Matrix
	// tiled_in is [output size X kernel size] X IC matrix. Each row is corresponding input values that multiplied with the kernel to produce one output.
	// weight is [kernel size X output channel] X IC matrix.
	// By multiplying tiled_in and weight, we can produce output values of each input channel.
	// Iterating as the number of input channels and accumulate all of the results, the convolution is done.
	// In the matrix multiplication, we set beta value as 1, so the dgemm does C = A x B + C, so accumulation is naturally done during matrix multiplication.

	int o_size = OW * OH, k_size = KW * KH;
	int r_idx, c_idx;
	double *tiled_in;	

        tiled_in = (double *)malloc((o_size * k_size * IC) * sizeof(double));

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

        for(int i = 0; i < IC; i++)
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				o_size, OC, k_size,
				1,
				tiled_in + i * k_size, k_size * IC,
				W + i * OC, OC * IC,
				1,
				O, OC);
		
        free(tiled_in);
}

void biasAdd(int size, int OC, double *I, double *B, double *O)
{
        for(int i = 0; i < size; i++)
        {
                cblas_daxpy(OC, 1, B, 1, O + i * OC, 1);
		cblas_daxpy(OC, 1, I + i * OC, 1, O + i * OC, 1);
        }
}

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
