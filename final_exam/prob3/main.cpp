#include <algorithm>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "conv.hpp"
#include "lib.hpp"

using namespace std;

struct Record {
	vector<float>					 result;
	map<string, chrono::nanoseconds> elapsed_map;
	int32_t							 shape[4];
	double							 precision;
	int								 testcase;
	string							 bit;
	double							 src_scale,
		weight_scale,
		dst_scale;
};

Numeric	 get_numeric_type(const char *type_name);
Record	 evaluate(const char *input_name, const char *weight_name, Numeric type);
ostream &operator<<(ostream &os, const Record &r);

int main(int argc, const char *argv[]) {
	if (argc < 4) {
		fprintf(stderr, "Usage: %s <input binary> <weight binary> [FP32|INT32|INT16]\n", argv[0]);
		exit(0);
	}

	const char *input_name	= argv[1];
	const char *weight_name = argv[2];
	Numeric		type		= get_numeric_type(argv[3]);

	auto record = evaluate(input_name, weight_name, type);

	cout << record << endl;

	Quintuple dst = {
		{record.shape[0], record.shape[1], record.shape[2], record.shape[3]},
		record.result.data(),
	};

	write_binary(dst, "output_tensor.bin");

	return 0;
}

Numeric get_numeric_type(const char *type_name) {
	static map<string, Numeric> m = {
		make_pair("32", Numeric::INT32),
		make_pair("16", Numeric::INT16),
		make_pair("8", Numeric::INT8),
		make_pair("FP32", Numeric::AVXFLOAT),
		make_pair("INT32", Numeric::AVXINT32),
		make_pair("INT16", Numeric::AVXINT16),
	};

	return m[type_name];
}

auto as_string = [](Numeric n) {
	static map<Numeric, string> m = {
		make_pair(Numeric::INT32, "int32"),
		make_pair(Numeric::INT16, "int16"),
		make_pair(Numeric::INT8, "int8"),
		make_pair(Numeric::AVXFLOAT, "avx_float"),
		make_pair(Numeric::AVXINT32, "avx_int32"),
		make_pair(Numeric::AVXINT16, "avx_int16"),
	};
	return m[n];
};

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

size_t get_bit_len(Numeric type) {
	switch (type) {
	case Numeric::FLOAT:
	case Numeric::INT32:
	case Numeric::AVXFLOAT:
	case Numeric::AVXINT32:
		return 32;
	case Numeric::INT16:
	case Numeric::AVXINT16:
		return 16;
	case Numeric::INT8:
		return 8;
	default:
		throw runtime_error("Unreachable code");
	}
}

Record evaluate(const char *input_name, const char *weight_name, Numeric type) {

	Quintuple input	 = read_binary(input_name);	 // batch, row, col, chan
	Quintuple weight = read_binary(weight_name); // row, col, out_chan, chan

	float *tmp = weight.arr;
	weight	   = transpose_channel(weight); // row, col, chan, out_chan
	free(tmp);

	int32_t dst_shape[] = {input.shape[0], input.shape[1], input.shape[2], weight.shape[3]};
	int32_t dst_len		= to_len(dst_shape);

	int32_t stride[] = {1, 1};
	float * expected = Conv2d<float>().calculate(input.arr, input.shape, weight.arr, weight.shape, stride, SAME);

	double src_scale	= get_optimal_scale(input.arr, to_len(input.shape), get_bit_len(type)),
		   weight_scale = get_optimal_scale(weight.arr, to_len(weight.shape), get_bit_len(type));

	auto   quantized = Quantizer(src_scale, weight_scale, type);
	float *actual	 = quantized.conv2d(input.arr, input.shape, weight.arr, weight.shape, stride, SAME);

	ssize_t len		  = input.shape[0] * input.shape[1] * input.shape[2] * weight.shape[3];
	auto	precision = get_precision(expected, actual, len);

	auto vec_result	 = vector<float>(actual, actual + dst_len);
	auto elapsed_map = timer.get_elapsed_map();
	timer.clear();

	free(expected);
	free(input.arr);
	free(weight.arr);
	free(actual);

	return Record{
		vec_result,
		elapsed_map,
		{dst_shape[0], dst_shape[1], dst_shape[2], dst_shape[3]},
		precision,
		get_testcase(input_name),
		as_string(type),
		src_scale,
		weight_scale,
	};
}

ostream &operator<<(ostream &os, const Record &r) {
	auto to_milli = [](auto i) {
		return (float) i / 1e6;
	};
	os << "testcase, " << r.testcase << ", "
	   << "numeric type, " << r.bit << ", "
	   << "scale, " << r.src_scale << ", " << r.weight_scale << ", ";
	for (auto i : r.elapsed_map) {
		os << i.first << ", " << to_milli(i.second.count()) << " ms, ";
	}
	os << "precision, " << r.precision * 100 << "%";

	return os;
}
