#!/bin/bash

make

for i in $(seq 3)
do
	./convolution ../test/$i/input_tensor.bin ../test/$i/kernel_tensor.bin || exit 255
done

make clean
