#include "lib.h"

#include <assert.h>
#include <stdlib.h>

void Fread(void *addr, ssize_t size, ssize_t n, FILE *fp) {
	ssize_t readn = fread(addr, size, n, fp);
	assert(readn == n);
}

Quintuple transpose_channel(Quintuple src) {
	int n_row	 = src.shape[0],
		n_col	 = src.shape[1],
		out_chan = src.shape[2],
		n_chan	 = src.shape[3];

	int col_len = n_chan * out_chan,
		row_len = col_len * n_col;

	Quintuple dst = {
		.shape = {n_row, n_col, n_chan, out_chan},
		.arr   = malloc(sizeof(float) * n_row * n_col * n_chan * out_chan),
	};

	for (int r = 0; r < n_row; ++r) {
		for (int c = 0; c < n_col; ++c) {
			for (int ch = 0; ch < n_chan; ++ch) {
				for (int o_ch = 0; o_ch < out_chan; ++o_ch) {
					dst.arr[r * row_len + c * col_len + ch * out_chan + o_ch] = src.arr[r * row_len + c * col_len + o_ch * n_chan + ch];
				}
			}
		}
	}

	return dst;
}

void write_binary(Quintuple record, const char *name) {
	FILE *fp = fopen(name, "wb+");
	fwrite(record.shape, sizeof(int32_t), 4, fp);
	int arr_len = record.shape[0] * record.shape[1] * record.shape[2] * record.shape[3];

	fwrite(record.arr, sizeof(float), arr_len, fp);

	fclose(fp);
}

Quintuple read_binary(const char *filename) {

	FILE *	  fp = fopen(filename, "rb");
	Quintuple record;

	Fread(record.shape, sizeof(int32_t), 4, fp);

	int64_t arr_len = record.shape[0] * record.shape[1] * record.shape[2] * record.shape[3];

	record.arr = malloc(sizeof(float) * arr_len);
	Fread(record.arr, sizeof(float), arr_len, fp);

	fclose(fp);
	return record;
}
