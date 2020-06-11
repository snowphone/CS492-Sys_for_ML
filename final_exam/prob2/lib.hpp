#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>

#include "type_enum.hpp"

#ifdef __cplusplus
#define restrict __restrict__
#endif

using namespace std;

template <typename Precision>
class Timer;

extern Timer<chrono::nanoseconds> timer;

#define errx(ERR, ...)            \
	fprintf(stderr, __VA_ARGS__); \
	exit(ERR)

typedef struct Quintuple {
	int32_t shape[4];
	float * arr;
} Quintuple;

void	  Fread(void *addr, ssize_t size, ssize_t n, FILE *fp);
Quintuple read_binary(const char *filename);
void	  write_binary(Quintuple record, const char *name);
Quintuple transpose_channel(Quintuple src);
double	  get_precision(float *restrict expected, float *restrict actual, ssize_t len);
double	  get_optimal_scale(float *restrict arr, ssize_t len, size_t bit_len);

template <typename Precision>
class Timer {
	map<string, chrono::time_point<chrono::system_clock>> timer_map;
	map<string, Precision>								  elapsed_map;

public:
	void begin_timer(const string &tag) {
		auto t_beg	   = chrono::system_clock::now();
		timer_map[tag] = t_beg;
	}

	void end_timer(const string &tag) {
		if (!timer_map.count(tag))
			throw runtime_error("TAG: '" + tag + "' does not exist");

		auto t_end		 = chrono::system_clock::now();
		auto t_beg		 = timer_map[tag];
		elapsed_map[tag] = chrono::duration_cast<Precision>(t_end - t_beg);
	}

	auto get_elapsed_map() const {
		return elapsed_map;
	}

	void clear() {
		timer_map.clear();
		elapsed_map.clear();
	}
};
