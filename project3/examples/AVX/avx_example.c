/*
 * Reference:
 * https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX
 */

#include <immintrin.h>
#include <stdio.h>

__m256 add(float* l, float* r) {
	__m256 lhs = _mm256_loadu_ps(l);
	__m256 rhs = _mm256_loadu_ps(r);
	return _mm256_add_ps(lhs, rhs);
}

int main()
{
    // Multiply 8 floats at a time
	float f_evens[] = {2, 4, 6, 8, 10, 12, 14, 16};
	float f_odds[]  = {15, 13, 11, 9, 7, 5, 3, 1};


    printf("evens: ");
    for (int i = 0; i < 8; i++) printf("%3.0f ", f_evens[i]);
    printf("\nodds:  ");
    for (int i = 0; i < 8; i++) printf("%3.0f ", f_odds[i]);

	__m256 ret = add(f_evens, f_odds);
	float res[8];
	_mm256_storeu_ps(res, ret);
    printf("\nres:   ");
    for (int i = 0; i < 8; i++) printf("%3.0f ", *(float *)&res[i]);
    printf("\n");

    return 0;
}
