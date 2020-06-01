#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <math.h>

void conv2D(int PW, int PH, int KW, int KH, int IC, int OC, int SW, int SH, int OW, int OH, double *I, double *W, double *O)
{
	// openBLAS implementation of 2D convolution
	// Simillar approach with Toepliz Matrix.
	// tiled_in is [output size X IC * kernel size] matrix. Each row is corresponding input values that multiplied with the kernel to produce one output.
	// weight is [IC * kernel size X output channel] matrix.
	// By multiplying tiled_in and weight, we can produce output values.
	// By calling the dgemm function once, we can perform convolution. The dgemm does Output = Tiled_input X weight.

	int o_size = OW * OH, k_size = KW * KH;
	int r_idx, c_idx;
	double *tiled_in;	

        tiled_in = (double *)malloc((o_size * k_size * IC) * sizeof(double));

        for(int ow = 0; ow < OW; ow++)
                for(int oh  = 0; oh < OH; oh++)
                {
                        r_idx = ow * OH + oh;
                        for(int i = 0; i < KW; i++)
                        	for(int j = 0; j < KH; j++)
                                {
                                	c_idx = i * KH + j;
					for(int ic = 0; ic < IC; ic++)
                                                tiled_in[(r_idx * k_size + c_idx) * IC + ic] = I[((ow * SW * PH + oh * SH) + (i * PH + j)) * IC + ic];
                               	}
                }

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			o_size, OC, k_size * IC,
			1,
			tiled_in, k_size * IC,
			W, OC,
			0,
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
