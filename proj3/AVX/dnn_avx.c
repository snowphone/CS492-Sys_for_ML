#include <stdio.h>
#include <stdint.h>
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
