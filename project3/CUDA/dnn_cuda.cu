#include <cassert>
#include <err.h>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <immintrin.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cuda_runtime.h>

extern "C" {

enum { avx2_sz = 8 };
typedef enum {
	VALID,
	SAME,
	POOLING,
	CONVOLVING,
} padding_t;

#define max(x, y) ((x) > (y) ? (x) : (y))

typedef struct Pool {
	int	  n_worker;
	pthread_t threads[];
} Pool;

typedef struct {
	float *arr;
	int	   size;
} Array;

typedef struct Queue {
	float *			iter;
	float *			end;
	pthread_mutex_t lock;
} Queue;

typedef struct Maxpool {
	float *const	 arr;
	int *const		 shape;
	const int *const ksize;
	const int *const strides;
	padding_t		 padding;
} Maxpool;

typedef struct mp_record_t {
	float *src_tensor;
	float *dst_pixel;
	int	   r;
	int	   c;
} mp_record_t;

typedef struct MPQueue {
	mp_record_t *	iter;
	mp_record_t *	end;
	pthread_mutex_t lock;
} MPQueue;


static void *leaky_relu_impl(void *_arg);

static void *batch_norm_impl(void *_args);
static void	 batch_normalization_pixel(float *const pixel,
									   const int	shape[],
									   float *const mean,
									   float *const var,
									   float *const gamma,
									   const float	epsilon);

static void *bias_add_impl(void *_args);
static void	 bias_add_pixel(float *const arr, int shape[], float *const biases);

static void *maxpool_impl(void *_args);
static void	 maxpool_receptive_field(
	 float *   src_tensor,
	 float *   dst,
	 int	   r,
	 int	   c,
	 const int shape[],
	 const int ksize[]);

static void	 matmul(const float * receptive_field,
					const float * weight,
					float *		  dst,

					const int n_strides,
					const int kernel_len,
					const int dst_chan);

static Pool *pool = NULL;

static int get_num_cpus() {
	return sysconf(_SC_NPROCESSORS_ONLN);
}

static Pool *init_pool() {
	const int numCPUs = get_num_cpus();
	Pool *	  pool	  = (Pool*)malloc(sizeof pool + numCPUs * sizeof(pthread_t));
	pool->n_worker	  = numCPUs;

	return pool;
}

/**
 * Deploy a thread pool and block until the job is completed.
 *
 * @param start_routine the function that takes void* argument
 * @param args An array of void pointers
 */
static void deploy(void *(*start_routine)(void *), void *args[]) {
	if (!pool)
		pool = init_pool();
	for (int i = 0; i < pool->n_worker; ++i) {
		pthread_create(pool->threads + i, NULL, start_routine, args);
	}

	for (int i = 0; i < pool->n_worker; ++i) {
		pthread_join(pool->threads[i], NULL);
	}
}

/**
 * A simple wrapper of mmap, which allocate in page-unit.
 */
static void *Mmap(const size_t size) {
	void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (addr == (void *) -1) {
		errx(ENOMEM, "Failed to allocate memory");
	}
	return addr;
}

/**
 * In-place version of leaky relu activation.
 *
 * @param[in,out] arr represented as 1-d array.
 * @param[in] size
 */
void leaky_relu(float *const arr, const int size) {
	if (!pool)
		pool = init_pool();

	const int job_size = size / pool->n_worker;
	float *	  arr_it   = arr;

	Array args[pool->n_worker];
	for (int i = 0; i < pool->n_worker; ++i) {
		if (i + 1 != pool->n_worker) {
			args[i].arr	 = arr_it;
			args[i].size = job_size;

			arr_it += job_size;
		} else {
			// Last
			args[i].arr	 = arr_it;
			args[i].size = arr + size - arr_it;

			arr_it += args[i].size;
		}
		pthread_create(pool->threads + i, NULL, leaky_relu_impl, args + i);
	}

	for (int i = 0; i < pool->n_worker; ++i) {
		pthread_join(pool->threads[i], NULL);
	}
}

static void *leaky_relu_impl(void *_arg) {
	Array *		 arg  = (Array*)_arg;
	float *const arr  = arg->arr;
	const int	 size = arg->size;

	const float *const src_end = arr + size;

	__m256 alpha = _mm256_set_ps(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
	float *it;
	for (it = arr; it + avx2_sz < src_end; it += avx2_sz) {
		__m256 v_src = _mm256_loadu_ps(it);

		__m256 alpha_src = _mm256_mul_ps(alpha, v_src);
		__m256 v_max	 = _mm256_max_ps(v_src, alpha_src);

		_mm256_storeu_ps(it, v_max);
	}

	// Not aligned to 32 bytes (8 floats)
	for (; it < src_end; ++it) {
		*it = max(*it, *it * 0.1);
	}

	return NULL;
}

void batch_normalization(float *const arr, const int shape[], float *const mean, float *const var, float *const gamma, const float epsilon) {

	float *			  end	 = arr + shape[0] * shape[1] * shape[2] * shape[3];
	Queue			  queue	 = {arr, end, PTHREAD_MUTEX_INITIALIZER};
	const void *const args[] = {(void*)&queue, (void*)shape, (void*)mean, (void*)var, (void*)gamma, (void*)&epsilon};

	deploy(batch_norm_impl, (void **) args);
}

static void *batch_norm_impl(void *_args) {
	void **args	   = (void**)_args;
	Queue *q	   = (Queue*)args[0];
	int *  shape   = (int*)args[1];
	float *mean	   = (float*)args[2];
	float *var	   = (float*)args[3];
	float *gamma   = (float*)args[4];
	float  epsilon = *(float *) args[5];

	while (1) {
		pthread_mutex_lock(&q->lock);
		float *const pixel = q->iter;
		q->iter += shape[3];
		pthread_mutex_unlock(&q->lock);

		if (pixel >= q->end)
			break;

		batch_normalization_pixel(pixel, shape, mean, var, gamma, epsilon);
	}

	return NULL;
}

static void batch_normalization_pixel(float *const pixel,
									  const int	   shape[],
									  float *const mean,
									  float *const var,
									  float *const gamma,
									  const float  epsilon) {
	const int n_chan = shape[3];
	/* Formula:
     x = (matrix - self.mean) / np.sqrt(self.variance + self.epsilon)
     self.result = self.gamma * x + self.beta
     */

	__m256 e  = _mm256_set1_ps(epsilon);
	float *it = pixel;
	int	   ch = 0;
	for (ch = 0; ch < n_chan; ch += avx2_sz) {
		__m256 x = _mm256_loadu_ps(it + ch);
		__m256 m = _mm256_loadu_ps(mean + ch);
		__m256 v = _mm256_loadu_ps(var + ch);
		__m256 g = _mm256_loadu_ps(gamma + ch);
		__m256 tmp;

		x		   = _mm256_sub_ps(x, m);
		tmp = _mm256_sqrt_ps(_mm256_add_ps(v, e));
		x		   = _mm256_div_ps(_mm256_mul_ps(g, x), tmp);

		_mm256_storeu_ps(it + ch, x);
	}

	for (; ch < n_chan; ++ch) {
		float x = it[ch];
		x		= gamma[ch] * (x - mean[ch]) / sqrt(var[ch] + epsilon);
		it[ch]	= x;
	}

	return;
}

void bias_add(float *const arr, int shape[], float *const biases) {
	const int size	 = shape[0] * shape[1] * shape[2] * shape[3];
	Queue	  queue	 = {arr, arr + size, PTHREAD_MUTEX_INITIALIZER};
	void *	  args[] = {(void*)&queue, (void*)shape, (void*)biases};

	deploy(bias_add_impl, (void **) args);
}

static void *bias_add_impl(void *_args) {
	void **args	  = (void**)_args;
	Queue *q	  = (Queue*)args[0];
	int *  shape  = (int*) args[1];
	float *biases = (float*)args[2];

	static bool is_set_len = false;
	static int	addr_len;
	if (!is_set_len) {
		addr_len   = (q->end - q->iter) / pool->n_worker / shape[3];
		is_set_len = true;
	}
	float *addresses[addr_len];
	memset(addresses, 0, sizeof addresses);
	int len = 0;

	while (1) {
		pthread_mutex_lock(&q->lock);
		for (int i = 0; i < addr_len; ++i) {
			addresses[len++] = q->iter;
			q->iter += shape[3];
			if (q->iter >= q->end)
				break;
		}
		pthread_mutex_unlock(&q->lock);

		for (int i = 0; i < len; ++i) {
			float *addr = addresses[i];
			if (addr && addr < q->end)
				bias_add_pixel(addr, shape, biases);
			else if (i == addr_len - 1)
				break;
			else
				goto exit;
		}
		len = 0;
	}
exit:
	return NULL;
}

static void bias_add_pixel(float *const arr, int shape[], float *const biases) {
	const int n_chan = shape[3];

	int ch = 0;
	for (ch = 0; ch < n_chan - avx2_sz; ch += avx2_sz) {
		__m256 x = _mm256_loadu_ps(arr + ch);
		__m256 b = _mm256_loadu_ps(biases + ch);
		_mm256_storeu_ps(arr + ch, _mm256_add_ps(x, b));
	}

	for (; ch < n_chan; ++ch) {
		arr[ch] = arr[ch] + biases[ch];
	}
}

static Maxpool create_maxpool_arr(Maxpool conv) {
	const int n_batch = conv.shape[0],
			  out_r	  = (conv.shape[1] - conv.ksize[0]) / conv.strides[0] + 1,
			  out_c	  = (conv.shape[2] - conv.ksize[1]) / conv.strides[1] + 1,
			  n_chan  = conv.shape[3],
			  size	  = n_batch * out_r * out_c * n_chan;

	Maxpool ret = {
		.arr	 = (float*) Mmap(sizeof(float) * size),
		.shape	 = (int*)malloc(sizeof(int) * 4),
		.ksize	 = conv.ksize,
		.strides = conv.strides,
		.padding = conv.padding,
	};

	int new_shape[] = {n_batch, out_r, out_c, n_chan};
	memmove(ret.shape, new_shape, sizeof(int) * 4);

	return ret;
}

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
			.arr	 = (float*)Mmap(sizeof(float) * new_size),
			.shape	 = (int*)malloc(sizeof(int) * 4),
			.ksize	 = src.ksize,
			.strides = src.strides,
			.padding = src.padding,
		};

		int new_shape[] = {n_batch, new_r, new_c, n_chan};
		memmove(padded.shape, new_shape, sizeof(int) * 4);

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
				.arr	 = (float*)Mmap(sizeof(float) * size),
				.shape	 = (int*)malloc(sizeof(int) * 4),
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
				.arr	 = (float*)Mmap(sizeof(float) * padded_size),
				.shape	 = (int*)malloc(sizeof(int) * 4),
				.ksize	 = src.ksize,
				.strides = src.strides,
				.padding = src.padding,
			};

			int new_shape[] = {n_batch, padded_row, padded_col, n_chan};
			memmove(dst.shape, new_shape, sizeof(int) * 4);

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

float *maxpool(const float src[],
			   const int   shape[],
			   const int   ksize[],
			   const int   strides[],
			   padding_t   padding) {
	Maxpool src_conv   = {(float *) src, (int *) shape, ksize, strides, padding};
	Maxpool padded_src = pad(src_conv, POOLING); // Only right and bottom are padded.
	Maxpool dst		   = create_maxpool_arr(padded_src);

	// max pooling
	const int n_batch = padded_src.shape[0],
			  n_chan  = padded_src.shape[3],
			  n_row	  = src_conv.shape[1],
			  n_col	  = src_conv.shape[2],

			  padded_row = padded_src.shape[1],
			  padded_col = padded_src.shape[2],

			  padded_size = n_batch * padded_row * padded_col * n_chan;

	int record_list_len = dst.shape[0] * dst.shape[1] * dst.shape[2] * dst.shape[3];

	mp_record_t *record_list = (mp_record_t*)Mmap(sizeof *record_list * record_list_len),
				*record_it	 = record_list;

	float *dst_pixel = dst.arr;
	for (int b = 0; b < n_batch; ++b) {
		float *const src_tensor = padded_src.arr + (b * padded_row * padded_col * n_chan);

		for (int r = 0; r < n_row; r += strides[0]) {
			for (int c = 0; c < n_col; c += strides[1], dst_pixel += n_chan) {
				if (!(r != n_row && c != n_col &&
					  r + ksize[0] <= padded_row && c + ksize[1] <= padded_col))
					continue;

				*record_it++ = (mp_record_t){src_tensor, dst_pixel, r, c};
			}
		}
	}
	MPQueue queue  = {record_list, record_it, PTHREAD_MUTEX_INITIALIZER};
	void *	args[] = {&queue, padded_src.shape, (void *) ksize};

	deploy(maxpool_impl, args);

	munmap(record_list, sizeof *record_list * record_list_len);
	munmap(padded_src.arr, padded_size);
	free(padded_src.shape);
	free(dst.shape);

	return dst.arr;
}

static void *maxpool_impl(void *_args) {
	void **args = (void**)_args;

	MPQueue *q		 = (MPQueue*)args[0];
	int *	 shape	 = (int*)args[1];
	int *	 ksize	 = (int*)args[2];

	while (true) {
		pthread_mutex_lock(&q->lock);
		mp_record_t *addr = q->iter;
		q->iter++;
		pthread_mutex_unlock(&q->lock);

		if (addr < q->end) {
			float * src_tensor = addr->src_tensor;
			float * dst_pixel  = addr->dst_pixel;
			int64_t r		   = addr->r;
			int64_t c		   = addr->c;
			maxpool_receptive_field(src_tensor, dst_pixel, r, c, shape, ksize);
		} else {
			break;
		}
	}

	return NULL;
}

static void maxpool_receptive_field(
	float * src_tensor,
	float * dst_pixel,
	const int		r,
	const int		c,
	const int		shape[],
	const int		ksize[]) {

	const int padded_col = shape[2],
			  n_chan	 = shape[3],

			  src_line_size = padded_col * n_chan;

	// Traversing elements in a pre-feature map
	// Initialize to -INF

	__m256 neg_inf = _mm256_set1_ps(-INFINITY);

	int ch;
	for (ch = 0; ch + avx2_sz < n_chan; ch += avx2_sz) {
		_mm256_storeu_ps(dst_pixel + ch, neg_inf);
	}
	for (; ch < n_chan; ++ch) {
		dst_pixel[ch] = -INFINITY;
	}

	for (int i = r; i < r + ksize[0]; ++i) {
		for (int j = c; j < c + ksize[1]; ++j) {
			// Channel wise maxpool
			float *src_pixel = (src_tensor + i * src_line_size + j * n_chan);

			for (ch = 0; ch + avx2_sz < n_chan; ch += avx2_sz) {
				__m256 x = _mm256_loadu_ps(dst_pixel + ch);
				__m256 s = _mm256_loadu_ps(src_pixel + ch);
				_mm256_storeu_ps(dst_pixel + ch, _mm256_max_ps(x, s));
			}
			for (; ch < n_chan; ++ch) {
				dst_pixel[ch] = max(dst_pixel[ch], src_pixel[ch]);
			}
		}
	}
}

float *conv2d(
	float * src,
	const int		shape[],
	float * weight,
	const int		k_shape[],
	const int		strides[],
	padding_t		padding) {

	const int ksize[2]	 = {k_shape[0], k_shape[1]};
	Maxpool	  _src		 = {src, (int*)shape, ksize, strides, padding};
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

	float *dst = (float*)Mmap(sizeof(float) * dst_size);

	float *	  dst_iter	 = dst;
	float *	  src_tensor = padded_src.arr;
	const int kernel_len = k_shape[0] * k_shape[1] * n_chan,
			  n_strides	 = dst_row * dst_col;

	float *		 receptive_fields = (float*)Mmap(sizeof(float) * kernel_len * n_strides);

	for (int b = 0; b < n_batch; ++b, src_tensor += b * src_tensor_size) {
		float *field_it = receptive_fields;

		for (int r = 0; r < n_row; r += strides[0]) {
			for (int c = 0; c < n_col; c += strides[1]) {
				if (!(r != n_row && c != n_col &&
					  r + k_shape[0] <= padded_row && c + k_shape[1] <= padded_col))
					continue;

				// Copy each receptive field to 1-d array
				float *src_line = src_tensor + r * src_line_size + c * n_chan;
				for (int i = r; i < r + k_shape[0]; ++i) {
					const int line_len = k_shape[1] * n_chan;
					memmove(field_it, src_line, sizeof(float) * line_len);
					field_it += line_len;
					src_line += src_line_size;
				}

				// Calculate conv operation
			}
		}
		
		matmul(receptive_fields, weight, dst_iter, n_strides, kernel_len, dst_chan);
		dst_iter += dst_chan * n_strides;
	}

	munmap(receptive_fields, sizeof(float) * n_strides * kernel_len);
	munmap(padded_src.arr, sizeof(float) * n_batch * padded_row * padded_col * n_chan);
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

	matmul_impl<<<dimGrid, dimBlock>>>(d_rf, d_w, d_dst, n_strides, kernel_len, dst_chan);

	HANDLE_ERROR(cudaMemcpy(dst, d_dst, sizeof(float) * dst_size, cudaMemcpyDeviceToHost));

	cudaFree(d_rf);
	cudaFree(d_w);
	cudaFree(d_dst);
	return;
}
}

