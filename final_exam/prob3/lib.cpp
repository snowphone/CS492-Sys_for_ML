#include "lib.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

Timer<chrono::nanoseconds> timer;

void Fread(void *addr, ssize_t size, ssize_t n, FILE *fp) {
	ssize_t readn = fread(addr, size, n, fp);
	if (readn != n)
		throw std::runtime_error("Failed to read");
}

Quintuple transpose_channel(Quintuple src) {
	int32_t n_row	 = src.shape[0],
			n_col	 = src.shape[1],
			out_chan = src.shape[2],
			n_chan	 = src.shape[3];

	int32_t col_len = n_chan * out_chan,
			row_len = col_len * n_col;

	Quintuple dst = {
		.shape = {n_row, n_col, n_chan, out_chan},
		.arr   = (float *) malloc(sizeof(*dst.arr) * n_row * n_col * n_chan * out_chan),
	};

	for (int32_t r = 0; r < n_row; ++r) {
		for (int32_t c = 0; c < n_col; ++c) {
			for (int32_t ch = 0; ch < n_chan; ++ch) {
				for (int32_t o_ch = 0; o_ch < out_chan; ++o_ch) {
					dst.arr[r * row_len + c * col_len + ch * out_chan + o_ch] =
						src.arr[r * row_len + c * col_len + o_ch * n_chan + ch];
				}
			}
		}
	}

	return dst;
}

void write_binary(Quintuple record, const char *name) {
	FILE *fp = fopen(name, "wb+");
	fwrite(record.shape, sizeof(int32_t), 4, fp);
	int32_t arr_len = record.shape[0] * record.shape[1] * record.shape[2] * record.shape[3];

	fwrite(record.arr, sizeof(float), arr_len, fp);

	fclose(fp);
}

Quintuple read_binary(const char *filename) {
	FILE *	  fp = fopen(filename, "rb");
	Quintuple record;

	Fread(record.shape, sizeof(int32_t), 4, fp);

	int32_t arr_len = record.shape[0] * record.shape[1] * record.shape[2] * record.shape[3];

	record.arr = (float *) malloc(sizeof(float) * arr_len);
	Fread(record.arr, sizeof(float), arr_len, fp);

	fclose(fp);
	return record;
}

double get_precision(float *restrict expected, float *restrict actual, ssize_t len) {
	double acc = 0.;
	for (int i = 0; i < len; ++i) {
		acc += pow(expected[i] - actual[i], 2);
	}

	acc		  = sqrt(acc / len);
	auto pair = minmax_element(expected, expected + len);

	return acc / (*pair.second - *pair.first);
}

double get_optimal_scale(float *restrict arr, ssize_t len, size_t bit_len) {
	auto   p	   = minmax_element(arr, arr + len);
	double arr_min = *p.first,
		   arr_max = *p.second;

	int64_t q_max = 1LL << (bit_len - 1),
			q_min = -q_max;

	return (q_max - q_min) / (arr_max - arr_min);
}
