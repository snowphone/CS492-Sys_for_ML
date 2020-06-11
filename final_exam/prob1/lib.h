#pragma once

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MEASURE(PROCEDURE)                                              \
	clock_t c_begin = clock();                                          \
	PROCEDURE;                                                          \
	clock_t c_end		 = clock();                                     \
	double	elapsed_time = 1000.0 * (c_end - c_begin) / CLOCKS_PER_SEC; \
	fprintf(stderr, "%s: %fms\n", #PROCEDURE, elapsed_time)

typedef struct Quintuple {
	int32_t shape[4];
	float * arr;
} Quintuple;

void	  Fread(void *addr, ssize_t size, ssize_t n, FILE *fp);
Quintuple read_binary(const char *filename);
void	  write_binary(Quintuple record, const char *name);
Quintuple transpose_channel(Quintuple src);

