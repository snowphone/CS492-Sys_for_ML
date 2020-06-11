#include "conv.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "lib.hpp"

#ifdef __cplusplus
#define restrict __restrict__
#endif

using namespace std;
using namespace chrono;

using T=float;
using R=float;


static void matmul(const T *restrict receptive_field,
			const T *restrict weight,
			T *restrict		  dst,

			const int32_t n_strides,
			const int32_t kernel_len,
			const int32_t dst_chan);

static int32_t calc_new_pad(const int32_t n, const int32_t k, const int32_t s, padding_t pad_opt) {
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
Maxpool pad(Maxpool src, padding_t option) {
	switch (option) {
	case POOLING: {
		const int32_t pad_r	   = calc_new_pad(src.shape[1], src.ksize[0], src.strides[0], src.padding),
					  pad_c	   = calc_new_pad(src.shape[2], src.ksize[1], src.strides[1], src.padding),
					  n_batch  = src.shape[0],
					  new_r	   = src.shape[1] + pad_r,
					  new_c	   = src.shape[2] + pad_c,
					  n_chan   = src.shape[3],
					  new_size = n_batch * new_r * new_c * n_chan;

		Maxpool padded = {
			(T *) malloc(sizeof(T) * new_size),
			(int32_t *) malloc(sizeof(int32_t) * 4),
			src.ksize,
			src.strides,
			src.padding,
		};

		int32_t tmp_shape[] = {n_batch, new_r, new_c, n_chan};
		memmove(padded.shape, tmp_shape, sizeof tmp_shape);

		// Right pad
		const int32_t n_row			= src.shape[1],
					  n_col			= src.shape[2],
					  src_line_size = n_col * n_chan,
					  dst_line_size = new_c * n_chan;

		T *src_line = src.arr;
		T *dst_line = padded.arr;
		for (int32_t r = 0; r < n_row; ++r, src_line += src_line_size, dst_line += dst_line_size) {

			memmove(dst_line, src_line, sizeof(T) * src_line_size);

			for (int32_t c = src_line_size; c < dst_line_size; ++c) {
				dst_line[c] = std::numeric_limits<T>::min();
			}
		}

		// Bottom pad
		for (T *it = dst_line; it < padded.arr + new_size; ++it) {
			*it = (T) std::numeric_limits<T>::min();
		}

		return padded;
	}
	case CONVOLVING: {
		if (src.padding == VALID) {
			const int32_t n_batch = src.shape[0],
						  n_row	  = src.shape[1],
						  n_col	  = src.shape[2],
						  n_chan  = src.shape[3],
						  size	  = n_batch * n_row * n_col * n_chan;

			Maxpool dst = {
				(T *) malloc(sizeof(T) * size),
				(int32_t *) malloc(sizeof(int32_t) * 4),
				src.ksize,
				src.strides,
				src.padding,
			};
			memmove(dst.arr, src.arr, sizeof(T) * size);
			memmove(dst.shape, src.shape, sizeof(int32_t) * 4);

			return dst;
		} else {
			const int32_t n_batch = src.shape[0],
						  n_row	  = src.shape[1],
						  n_col	  = src.shape[2],
						  n_chan  = src.shape[3],

						  r_pad = ((src.strides[0] - 1) * n_row - 1 + src.ksize[0]) / 2,
						  c_pad = ((src.strides[1] - 1) * n_col - 1 + src.ksize[1]) / 2,

						  padded_row  = 2 * r_pad + n_row,
						  padded_col  = 2 * c_pad + n_col,
						  padded_size = n_batch * padded_row * padded_col * n_chan;

			Maxpool dst = {
				(T *) malloc(sizeof(T) * padded_size),
				(int32_t *) malloc(sizeof(int32_t) * 4),
				src.ksize,
				src.strides,
				src.padding,
			};

			int32_t tmp_shape[] = {n_batch, padded_row, padded_col, n_chan};
			memmove(dst.shape, tmp_shape, sizeof tmp_shape);

			const int32_t vertical_pad_len	 = c_pad * padded_col * n_chan,
						  horizontal_pad_len = r_pad * n_chan,
						  src_line_size		 = n_col * n_chan;

			T *dst_it = dst.arr;
			T *src_it = src.arr;
			for (int32_t b = 0; b < n_batch; ++b) {

				// Up
				memset(dst_it, 0, sizeof(T) * vertical_pad_len);
				dst_it += vertical_pad_len;

				for (int32_t r = 0; r < n_row; ++r) {
					// Left
					memset(dst_it, 0, sizeof(T) * horizontal_pad_len);
					dst_it += horizontal_pad_len;

					memmove(dst_it, src_it, sizeof(T) * src_line_size);

					dst_it += src_line_size;
					src_it += src_line_size;

					// Right
					memset(dst_it, 0, sizeof(T) * horizontal_pad_len);
					dst_it += horizontal_pad_len;
				}

				// Bottom
				memset(dst_it, 0, sizeof(T) * vertical_pad_len);
				dst_it += vertical_pad_len;
			}

			return dst;
		}
	}
	default:
		errx(EINVAL, "%s %d: Invalid argument", __func__, __LINE__);
	}
}


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
	padding_t	  padding) {

	const int32_t ksize[2]	 = {k_shape[0], k_shape[1]};
	Maxpool	  _src		 = {src, (int32_t *) shape, ksize, strides, padding};
	Maxpool	  padded_src = pad(_src, CONVOLVING);

	const int32_t n_batch = shape[0],
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

	R *dst = (R *) malloc(sizeof(R) * dst_size);

	R *			  dst_iter	 = dst;
	T *			  src_tensor = padded_src.arr;
	const int32_t kernel_len = k_shape[0] * k_shape[1] * n_chan,
		  n_strides = dst_row * dst_col;

	float* receptive_fields = (float*) malloc(sizeof(float) * kernel_len * n_strides);

	for (int32_t b = 0; b < n_batch; ++b, src_tensor += b * src_tensor_size) {
		float *field_it = receptive_fields;

		for (int32_t r = 0; r < n_row; r += strides[0]) {
			for (int32_t c = 0; c < n_col; c += strides[1]) {
				if (!(r != n_row && c != n_col &&
					  r + k_shape[0] <= padded_row && c + k_shape[1] <= padded_col))
					continue;

				// Copy each receptive field to 1-d array
				T *src_line = src_tensor + r * src_line_size + c * n_chan;
				for (int32_t i = r; i < r + k_shape[0]; ++i) {
					const int32_t line_len = k_shape[1] * n_chan;
					memmove(field_it, src_line, sizeof(T) * line_len);
					field_it += line_len;
					src_line += src_line_size;
				}

				// Calculate conv operation
				// weight: (dst_chan, ffc)
				// receptive field: (ffc, n_strides)
			}

		}
		matmul(receptive_fields, weight, dst_iter, n_strides, kernel_len, dst_chan);
		dst_iter += dst_chan * n_strides;
	}

	free(receptive_fields);
	free(padded_src.arr);
	free(padded_src.shape);
	return dst;
}



#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        errx(EXIT_FAILURE, "%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
    }
}


__global__ static void matmul_impl(const float * d_rf,
				   const float * d_w,
				   float *		 d_dst,

				   const int n_strides,
				   const int kernel_len,
				   const int dst_chan) {

	int row = blockIdx.y * blockDim.y +  threadIdx.y,
		col = blockIdx.x * blockDim.x +  threadIdx.x;


	if(row < n_strides && col < dst_chan) {
		float acc = 0.f;
		for (int k = 0; k < kernel_len; ++k) {
			acc += d_rf[row * kernel_len + k] * d_w[k * dst_chan + col];
		}
		d_dst[row * dst_chan + col] = acc;
	}
}

static void matmul(const float * receptive_field,
				   const float * weight,
				   float *		 dst,

				   const int n_strides,
				   const int kernel_len,
				   const int dst_chan) {
	const int rf_size  = n_strides * kernel_len,
			  w_size   = kernel_len * dst_chan,
			  dst_size = n_strides * dst_chan;
	float *d_rf, 
		  *d_w, 
		  *d_dst;

	HANDLE_ERROR(cudaMalloc((void**)&d_rf, sizeof(float) * rf_size));
	HANDLE_ERROR(cudaMalloc((void**)&d_w, sizeof(float) * w_size));
	HANDLE_ERROR(cudaMalloc((void**)&d_dst, sizeof(float) * dst_size));


	HANDLE_ERROR(cudaMemcpy(d_rf, receptive_field, sizeof(float) * rf_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_w, weight, sizeof(float) * w_size, cudaMemcpyHostToDevice));
	

	int blockSize = 16;
	dim3 dimBlock(blockSize, blockSize);
	// COL, ROW
	dim3 dimGrid(ceil((float)dst_chan / dimBlock.x), ceil((float)n_strides / dimBlock.y));

	timer.begin_timer("conv2d");
	matmul_impl<<<dimGrid, dimBlock>>>(d_rf, d_w, d_dst, n_strides, kernel_len, dst_chan);
	timer.end_timer("conv2d");

	HANDLE_ERROR(cudaMemcpy(dst, d_dst, sizeof(float) * dst_size, cudaMemcpyDeviceToHost));

	cudaFree(d_rf);
	cudaFree(d_w);
	cudaFree(d_dst);
	return;
}


