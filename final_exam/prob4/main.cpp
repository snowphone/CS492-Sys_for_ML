#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include "conv.hpp"
#include "lib.hpp"

using namespace std;

struct Record {
	vector<float>					 result;
	map<string, chrono::nanoseconds> elapsed_map;
	int32_t							 shape[4];
	int								 testcase;
	string							 bit;
};

Record	 evaluate(const char *input_name, const char *weight_name);
ostream &operator<<(ostream &os, const Record &r);

int main(int argc, const char *argv[]) {
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <input binary> <weight binary>\n", argv[0]);
		exit(0);
	}

	const char *input_name	= argv[1];
	const char *weight_name = argv[2];

	auto record = evaluate(input_name, weight_name);

	cout << record << endl;

	Quintuple dst = {
		{record.shape[0], record.shape[1], record.shape[2], record.shape[3]},
		record.result.data(),
	};

	write_binary(dst, "output_tensor.bin");

	return 0;
}

#ifndef RELEASE
auto get_testcase = [](string p) {
	if (p.find("/1/") != string::npos) {
		return 1;
	} else if (p.find("/2/") != string::npos) {
		return 2;
	} else if (p.find("/3/") != string::npos) {
		return 3;
	} else {
		throw std::runtime_error("Not valid path: " + p);
	}
};
#endif

Record evaluate(const char *input_name, const char *weight_name) {

	Quintuple input	 = read_binary(input_name);	 // batch, row, col, chan
	Quintuple weight = read_binary(weight_name); // row, col, out_chan, chan

	float *tmp = weight.arr;
	weight	   = transpose_channel(weight); // row, col, chan, out_chan
	free(tmp);

	auto get_len = [](int shape[4]) {
		return accumulate(shape, shape + 4, 1, [](auto l, auto r) { return l * r; });
	};

	int32_t dst_shape[] = {input.shape[0], input.shape[1], input.shape[2], weight.shape[3]};
	int32_t dst_len		= get_len(dst_shape);

	int32_t stride[] = {1, 1};
	float * dst		 = conv2d(input.arr, input.shape, weight.arr, weight.shape, stride, SAME);

	auto vec_result	 = vector<float>(dst, dst + dst_len);
	auto elapsed_map = timer.get_elapsed_map();

	free(dst);
	free(input.arr);
	free(weight.arr);

	return Record{
		vec_result,
		elapsed_map,
		{dst_shape[0], dst_shape[1], dst_shape[2], dst_shape[3]},
		get_testcase(input_name),
	};
}

ostream &operator<<(ostream &os, const Record &r) {
	auto to_milli = [](auto i) {
		return (float) i / 1e6;
	};
#ifndef RELEASE
	os << "testcase, " << r.testcase << ", ";
#endif
	for (auto i : r.elapsed_map) {
		os << i.first << ", " << to_milli(i.second.count()) << " ms";
	}

	return os;
}
