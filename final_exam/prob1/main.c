#include <stdio.h>
#include <stdlib.h>

#include "conv.h"
#include "lib.h"


int main(int argc, const char *argv[]) {
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <input binary> <weight binary>\n", argv[0]);
		exit(0);
	}

	const char *input_name	= argv[1];
	const char *weight_name = argv[2];

	Quintuple input	 = read_binary(input_name);	 // batch, row, col, chan
	Quintuple weight = read_binary(weight_name); // row, col, out_chan, chan
	float *	  tmp	 = weight.arr;
	weight			 = transpose_channel(weight); // row, col, chan, out_chan
	free(tmp);

	MEASURE(float *ret = conv2d(input.arr, input.shape, weight.arr, weight.shape, (int[]){1, 1}, SAME));

	Quintuple dst = {
		.shape = {input.shape[0], input.shape[1], input.shape[2], weight.shape[3]},
		.arr   = ret,
	};

	write_binary(dst, "output_tensor.bin");

	free(ret);
	free(input.arr);
	free(weight.arr);
	return 0;
}
