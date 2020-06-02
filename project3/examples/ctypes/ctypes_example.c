#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

void add_float_p(float a, float b, float *c) {
    *c = a + b;
}


float* double_vector(float* f, size_t size) {
	for(int i = 0; i < size; ++i) {
		f[i] *= 2;
	}

	return f;
}


void print(float* f, int size, int dim, int* shape) {
	fprintf(stderr, "(");
	for(int i = 0; i < dim; ++i) {
		fprintf (stderr, "%d ", shape[i]);
	}
	fprintf(stderr, ")");
}

float *double_arr(float *f, int size) {
	fprintf(stderr, __func__);
	float* new_ary = malloc(sizeof *f * size * 2);
	memmove(new_ary, f, size * sizeof *f);
	memmove(new_ary + size, f, size * sizeof *f);

	return new_ary;
}
