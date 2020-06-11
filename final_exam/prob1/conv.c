#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "conv.h"

#define errx(ERR, ...)            \
	fprintf(stderr, __VA_ARGS__); \
	exit(ERR)

#define max(x, y) ((x) > (y) ? (x) : (y))

typedef struct Maxpool {
	float *const	 arr;
	int *const		 shape;
	const int *const ksize;
	const int *const strides;
	padding_t		 padding;
} Maxpool;

static void matmul(const float *restrict receptive_field,
				   const float *restrict weight,
				   float *restrict		 dst,

				   const int n_strides,
				   const int kernel_len,
				   const int dst_chan);

static int calc_new_pad(const int n, const int k, const int s, padding_t pad_opt) {
	switch (pad_opt) {
	case VALID:
		return 0;
	case SAME:
		if (n % s)
			return max(k - (n % s), 0);
		else
			return max(k - s, 0);
	default:
		errx(EINVAL, "Only VALID or SAME enum is allowed");
	}
}

/**
 * Insert padding
 *
 * @param src
 * @param option If option is set to PADDING, -INFs are padded into the right and the bottom.
 * 				Otherwise (set to CONVOLVING), zero values are padded.
 */
static Maxpool pad(Maxpool src, padding_t option) {
	switch (option) {
	case POOLING: {
		const int pad_r	   = calc_new_pad(src.shape[1], src.ksize[0], src.strides[0], src.padding),
				  pad_c	   = calc_new_pad(src.shape[2], src.ksize[1], src.strides[1], src.padding),
				  n_batch  = src.shape[0],
				  new_r	   = src.shape[1] + pad_r,
				  new_c	   = src.shape[2] + pad_c,
				  n_chan   = src.shape[3],
				  new_size = n_batch * new_r * new_c * n_chan;

		Maxpool padded = {
			.arr	 = malloc(sizeof(float) * new_size),
			.shape	 = malloc(sizeof(int) * 4),
			.ksize	 = src.ksize,
			.strides = src.strides,
			.padding = src.padding,
		};

		memmove(padded.shape, (int[]){n_batch, new_r, new_c, n_chan}, sizeof(int) * 4);

		// Right pad
		const int n_row			= src.shape[1],
				  n_col			= src.shape[2],
				  src_line_size = n_col * n_chan,
				  dst_line_size = new_c * n_chan;
		float *src_line			= src.arr;
		float *dst_line			= padded.arr;
		for (int r = 0; r < n_row; ++r, src_line += src_line_size, dst_line += dst_line_size) {

			memmove(dst_line, src_line, sizeof(float) * src_line_size);

			for (int c = src_line_size; c < dst_line_size; ++c) {
				dst_line[c] = -INFINITY;
			}
		}

		// Bottom pad
		for (float *it = dst_line; it < padded.arr + new_size; ++it) {
			*it = -INFINITY;
		}

		return padded;
	}
	case CONVOLVING: {
		if (src.padding == VALID) {
			const int n_batch = src.shape[0],
					  n_row	  = src.shape[1],
					  n_col	  = src.shape[2],
					  n_chan  = src.shape[3],
					  size	  = n_batch * n_row * n_col * n_chan;

			Maxpool dst = {
				.arr	 = malloc(sizeof(float) * size),
				.shape	 = malloc(sizeof(int) * 4),
				.ksize	 = src.ksize,
				.strides = src.strides,
				.padding = src.padding,
			};
			memmove(dst.arr, src.arr, sizeof(float) * size);
			memmove(dst.shape, src.shape, sizeof(int) * 4);

			return dst;
		} else {
			const int n_batch = src.shape[0],
					  n_row	  = src.shape[1],
					  n_col	  = src.shape[2],
					  n_chan  = src.shape[3],

					  r_pad = ((src.strides[0] - 1) * n_row - 1 + src.ksize[0]) / 2,
					  c_pad = ((src.strides[1] - 1) * n_col - 1 + src.ksize[1]) / 2,

					  padded_row  = 2 * r_pad + n_row,
					  padded_col  = 2 * c_pad + n_col,
					  padded_size = n_batch * padded_row * padded_col * n_chan;

			Maxpool dst = {
				.arr	 = malloc(sizeof(float) * padded_size),
				.shape	 = malloc(sizeof(int) * 4),
				.ksize	 = src.ksize,
				.strides = src.strides,
				.padding = src.padding,
			};

			memmove(dst.shape, (int[]){n_batch, padded_row, padded_col, n_chan}, sizeof(int) * 4);

			const int vertical_pad_len	 = c_pad * padded_col * n_chan,
					  horizontal_pad_len = r_pad * n_chan,
					  src_line_size		 = n_col * n_chan;

			float *dst_it = dst.arr;
			float *src_it = src.arr;
			for (int b = 0; b < n_batch; ++b) {

				// Up
				memset(dst_it, 0, sizeof(float) * vertical_pad_len);
				dst_it += vertical_pad_len;

				for (int r = 0; r < n_row; ++r) {
					// Left
					memset(dst_it, 0, sizeof(float) * horizontal_pad_len);
					dst_it += horizontal_pad_len;

					memmove(dst_it, src_it, sizeof(float) * src_line_size);

					dst_it += src_line_size;
					src_it += src_line_size;

					// Right
					memset(dst_it, 0, sizeof(float) * horizontal_pad_len);
					dst_it += horizontal_pad_len;
				}

				// Bottom
				memset(dst_it, 0, sizeof(float) * vertical_pad_len);
				dst_it += vertical_pad_len;
			}

			return dst;
		}
	}
	default:
		errx(EINVAL, "%s %d: Invalid argument", __func__, __LINE__);
	}
}

float *conv2d(
	float *restrict src,
	const int		shape[static 4],
	float *restrict weight,
	const int		k_shape[static 4],
	const int		strides[static 2],
	padding_t		padding) {

	const int ksize[2]	 = {k_shape[0], k_shape[1]};
	Maxpool	  _src		 = {src, (void *) shape, ksize, strides, padding};
	Maxpool	  padded_src = pad(_src, CONVOLVING);

	const int n_batch = shape[0],
			  n_row	  = shape[1],
			  n_col	  = shape[2],
			  n_chan  = shape[3],

			  padded_row = padded_src.shape[1],
			  padded_col = padded_src.shape[2],

			  dst_row  = (padded_row - k_shape[0]) / strides[0] + 1,
			  dst_col  = (padded_col - k_shape[1]) / strides[1] + 1,
			  dst_chan = k_shape[3],
			  dst_size = n_batch * dst_row * dst_col * dst_chan,

			  src_tensor_size = padded_row * padded_col * n_chan,
			  src_line_size	  = padded_col * n_chan;

	float *dst = malloc(sizeof(float) * dst_size);

	float *	  dst_iter	 = dst;
	float *	  src_tensor = padded_src.arr;
	const int kernel_len = k_shape[0] * k_shape[1] * n_chan,
			  n_strides	 = dst_row * dst_col;

	for (int b = 0; b < n_batch; ++b, src_tensor += b * src_tensor_size) {

		for (int r = 0; r < n_row; r += strides[0]) {
			for (int c = 0; c < n_col; c += strides[1]) {
				if (!(r != n_row && c != n_col &&
					  r + k_shape[0] <= padded_row && c + k_shape[1] <= padded_col))
					continue;

				// Copy each receptive field to 1-d array
				float *src_line = src_tensor + r * src_line_size + c * n_chan;
				float  receptive_field[kernel_len];
				float *field_it = receptive_field;
				for (int i = r; i < r + k_shape[0]; ++i) {
					const int line_len = k_shape[1] * n_chan;
					memmove(field_it, src_line, sizeof(float) * line_len);
					field_it += line_len;
					src_line += src_line_size;
				}

				// Calculate conv operation
				// weight: (dst_chan, ffc)
				// receptive field: (ffc, n_strides)
				matmul(receptive_field, weight, dst_iter, 1, kernel_len, dst_chan);
				dst_iter += dst_chan;
			}
		}
	}

	free(padded_src.arr);
	free(padded_src.shape);
	return dst;
}

static void matmul(const float *restrict receptive_field,
				   const float *restrict weight,
				   float *restrict		 dst,

				   const int n_strides,
				   const int kernel_len,
				   const int dst_chan) {
	memset(dst, 0, sizeof(float) * n_strides * dst_chan);

	for (int j = 0; j < kernel_len; ++j) {
		float *rhs_line = (float *) &weight[j * dst_chan];

		for (int i = 0; i < n_strides; ++i) {
			float  x		= receptive_field[i * kernel_len + j];
			float *dst_line = &dst[i * dst_chan];

			for (int k = 0; k < dst_chan; ++k) {
				dst_line[k] += x * rhs_line[k];
			}
		}
	}
}
