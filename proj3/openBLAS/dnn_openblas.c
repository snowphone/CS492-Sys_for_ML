#include <stdio.h>
#include <cblas.h>

void openblas_sgemm(int M, int N, int K, float *A, float *B, float *C)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			M, N, K, 
			1,
			A, K,
			B, N,
			0,
			C, M
		   );
}

void openblas_dgemm(int M, int N, int K, double *A, double *B, double *C)
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

void conv2D(int PW, int PH, int KW, int KH, int IC, int OC, int SW, int SH, int OW, int OH, double *I, double *W, double *O)
{
	// for i in input channel
	// Consider stride I -> matrix multiple-able shape IS
	// matrix multiple IS X W + 1 * O = O
	int o_size = OW * OH, k_size = KW * KH;
	int r_idx, c_idx;

	for(int ic = 0; ic < IC; ic++)
	{
		double* matA = malloc((o_size * k_size) * sizeof(double));
			
		for(int ow = 0; i < OW; i++)
		{
			for(int oh  = 0; j < OH; j++)
			{
				r_idx = ow * OH + oh;
				for(int i = 0; i < KW; i++)
				{
					for(int j = 0; j < KH; j++)
					{
						c_idx = i * KH + j;
						matA[r_idx * k_size + c_idx] = I[(ow * SW * PH + oh * SH) + (i * PH  + j)];
					}
				}
			}
		}

		cblas_dgemm(CblasRowmajor, CblasNoTrans, CblasNoTrans,
				o_size, OC, k_size,
				1,
				matA, k_size,
				W + ic * (k_size * OC), OC,//depending on how W given.
				1,
				O, o_size
			   );
		free(matA);
	}
}
