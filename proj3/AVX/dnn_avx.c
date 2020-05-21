#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <immintrin.h>
#include <unistd.h>
#include <pthread.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

typedef const int cint;

typedef struct Workers {
  size_t n_worker;
  pthread_t threads[];
} Workers;

typedef void *(*func_t)(void *);

Workers* workers = NULL;

Workers *init_pool() {
  int numCPUs = sysconf(_SC_NPROCESSORS_ONLN);
  Workers *workers = malloc(sizeof workers + numCPUs * sizeof(pthread_t));
  workers->n_worker = numCPUs;


  return workers;
}

typedef struct {
	float* arr;
	int size;
} Relu_arg_t;

static void* leaky_relu_impl(void* _arg);

/**
 * In-place version of leaky relu activation.
 *
 * @param[in,out] arr represented as 1-d array.
 * @param[in] size
 */
void leaky_relu(float *const arr, const int size) {
	if(!workers)
          workers = init_pool();

	const int job_size = size / workers->n_worker;
	float* arr_it = arr;

	Relu_arg_t args[workers->n_worker];
	for(int i = 0; i < workers->n_worker; ++i) {
		if(i + 1 != workers->n_worker) {
			args[i].arr = arr_it;
			args[i].size = job_size;

			arr_it += job_size;
		} else {
			// Last
			args[i].arr = arr_it;
			args[i].size = arr + size - arr_it;

			arr_it += args[i].size;
		}
		pthread_create(workers->threads + i, NULL, leaky_relu_impl, args + i);
	}

	for(int i = 0; i < workers->n_worker; ++i) {
		pthread_join(workers->threads[i], NULL);
	}
}

static void* leaky_relu_impl(void* _arg) {
	Relu_arg_t* arg = _arg;
	float *const arr = arg->arr;
	const int size = arg->size;

	const float *const src_end = arr + size;

        __m256 alpha = _mm256_set_ps(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
	float *it;
	for (it = arr; it < src_end; it += 8) {
		__m256 v_src = _mm256_loadu_ps(it);

		__m256 alpha_src = _mm256_mul_ps(alpha, v_src);
		__m256 v_max = _mm256_max_ps(v_src, alpha_src);

		_mm256_storeu_ps(it, v_max);
	}

	if (it != src_end) {
		// Not aligned to 32 bytes (8 floats)
		for (it -= 8; it < src_end; ++it) {
			*it = MAX(*it, *it * 0.1);
		}
	}

	return NULL;
}
