#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <immintrin.h>
#include <pthread.h>
#include <unistd.h>

enum { avx2_sz = 8 };

#define max(x, y) ((x) > (y) ? (x) : (y))

typedef const int cint;

typedef struct Pool {
	size_t n_worker;
	pthread_t threads[];
} Pool;

typedef struct {
	float *arr;
	int size;
} Array;

typedef struct Queue {
	float *iter;
	float *end;
	pthread_mutex_t lock;
} Queue;

static void *leaky_relu_impl(void *_arg);
static void *batch_norm_impl(void *_args);
static void batch_normalization_pixel(float *const pixel,
									  const int shape[static 4],
									  float *const mean, float *const var,
									  float *const gamma, const float epsilon);
static void *bias_add_impl(void *_args);
static void bias_add_pixel(float *const arr, int shape[static 4],
						   float *const biases);

static Pool *pool = NULL;

static Pool *init_pool() {
	const int numCPUs = sysconf(_SC_NPROCESSORS_ONLN);
	Pool *pool = malloc(sizeof pool + numCPUs * sizeof(pthread_t));
	pool->n_worker = numCPUs;

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
 * In-place version of leaky relu activation.
 *
 * @param[in,out] arr represented as 1-d array.
 * @param[in] size
 */
void leaky_relu(float *const arr, const int size) {
	if (!pool)
		pool = init_pool();

	const int job_size = size / pool->n_worker;
	float *arr_it = arr;

	Array args[pool->n_worker];
	for (int i = 0; i < pool->n_worker; ++i) {
		if (i + 1 != pool->n_worker) {
			args[i].arr = arr_it;
			args[i].size = job_size;

			arr_it += job_size;
		} else {
			// Last
			args[i].arr = arr_it;
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
	Array *arg = _arg;
	float *const arr = arg->arr;
	const int size = arg->size;

	const float *const src_end = arr + size;

	__m256 alpha = _mm256_set_ps(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
	float *it;
	for (it = arr; it < src_end; it += avx2_sz) {
		__m256 v_src = _mm256_loadu_ps(it);

		__m256 alpha_src = _mm256_mul_ps(alpha, v_src);
		__m256 v_max = _mm256_max_ps(v_src, alpha_src);

		_mm256_storeu_ps(it, v_max);
	}

	if (it != src_end) {
		// Not aligned to 32 bytes (8 floats)
		for (it -= avx2_sz; it < src_end; ++it) {
			*it = max(*it, *it * 0.1);
		}
	}

	return NULL;
}

void batch_normalization(float *const arr, const int shape[static 4],
						 float *const mean, float *const var,
						 float *const gamma, const float epsilon) {

	float *end = arr + shape[0] * shape[1] * shape[2] * shape[3];
	Queue queue = {arr, end, PTHREAD_MUTEX_INITIALIZER};
	const void *const args[] = {&queue, shape, mean, var, gamma, &epsilon};

	deploy(batch_norm_impl, (void *)args);
}

static void *batch_norm_impl(void *_args) {
	void **args = _args;
	Queue *q = args[0];
	int *shape = args[1];
	float *mean = args[2];
	float *var = args[3];
	float *gamma = args[4];
	float epsilon = *(float *)args[5];

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
									  const int shape[static 4],
									  float *const mean, float *const var,
									  float *const gamma, const float epsilon) {
	const int n_batch = shape[0], n_row = shape[1], n_col = shape[2],
			  n_chan = shape[3];
	const int chan_size = 1, col_size = n_chan * chan_size,
			  row_size = n_col * col_size, batch_size = n_row * row_size,
			  size = n_batch * batch_size;
	/* Formula:
	   x = (matrix - self.mean) / np.sqrt(self.variance + self.epsilon)
	   self.result = self.gamma * x + self.beta
	   */

	__m256 e = _mm256_set_ps(epsilon, epsilon, epsilon, epsilon, epsilon,
							 epsilon, epsilon, epsilon);
	float *it = pixel;
	int ch = 0;
	for (ch = 0; ch < n_chan; ch += avx2_sz) {
		__m256 x = _mm256_loadu_ps(it + ch);
		__m256 m = _mm256_loadu_ps(mean + ch);
		__m256 v = _mm256_loadu_ps(var + ch);
		__m256 g = _mm256_loadu_ps(gamma + ch);

		x = _mm256_sub_ps(x, m);
		__m256 tmp = _mm256_sqrt_ps(_mm256_add_ps(v, e));
		x = _mm256_div_ps(_mm256_mul_ps(g, x), tmp);

		_mm256_storeu_ps(it + ch, x);
	}

	for (; ch < n_chan; ++ch) {
		float x = it[ch];
		x = gamma[ch] * (x - mean[ch]) / sqrt(var[ch] + epsilon);
		it[ch] = x;
	}

	return;
}

void bias_add(float *const arr, int shape[static 4], float *const biases) {
	const int size = shape[0] * shape[1] * shape[2] * shape[3];
	Queue queue = {arr, arr + size, PTHREAD_MUTEX_INITIALIZER};
	void *args[] = {&queue, shape, biases};

	deploy(bias_add_impl, (void *)args);
}

static void *bias_add_impl(void *_args) {
	void **args = _args;
	Queue *q = args[0];
	int *shape = args[1];
	float *biases = args[2];

	while (1) {
		pthread_mutex_lock(&q->lock);
		float *arr = q->iter;
		q->iter += shape[3];
		pthread_mutex_unlock(&q->lock);

		if (arr < q->end)
			bias_add_pixel(arr, shape, biases);
		else
			break;
	}

	return NULL;
}

static void bias_add_pixel(float *const arr, int shape[static 4],
						   float *const biases) {
	const int size = shape[0] * shape[1] * shape[2] * shape[3],
			  n_chan = shape[3];
	const float *const end = arr + size;

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

