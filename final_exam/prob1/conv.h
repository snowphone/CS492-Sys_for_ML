#pragma once

typedef enum {
	VALID,
	SAME,
	POOLING,
	CONVOLVING,
} padding_t;

/**
 * @param src
 * @param shape {n_batch, n_row, n_col, n_chan}
 * @param weight
 * @param k_shape {k_row, k_col, in_chan, dst_chan}
 * @param strides {row, col}
 * @param padding  VALID | SAME
 */
float *conv2d(
	float *restrict src,
	const int		shape[static 4],
	float *restrict weight,
	const int		k_shape[static 4],
	const int		strides[static 2],
	padding_t		padding);
