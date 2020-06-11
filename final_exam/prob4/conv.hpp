#pragma once

#include <cstdint>


#ifdef __cplusplus
#define restrict __restrict__
#endif

using namespace std;

using T=float;
using R=float;

typedef enum {
	VALID,
	SAME,
	POOLING,
	CONVOLVING,
} padding_t;

struct Maxpool {
	T *const			 arr;
	int32_t *const		 shape;
	const int32_t *const ksize;
	const int32_t *const strides;
	padding_t			 padding;
};


/**
 * @param src
 * @param shape {n_batch, n_row, n_col, n_chan}
 * @param weight
 * @param k_shape {k_row, k_col, in_chan, dst_chan}
 * @param strides {row, col}
 * @param padding  VALID | SAME
 */
R *conv2d(
	T *restrict	  src,
	const int32_t shape[4],
	T *restrict	  weight,
	const int32_t k_shape[4],
	const int32_t strides[2],
	padding_t	  padding);

