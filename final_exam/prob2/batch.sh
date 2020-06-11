#!/bin/bash


make clean && make

for bit in 8 16 32
do
	for i in $(seq 3)
	do
		./convolution ../test/$i/input_tensor.bin ../test/$i/kernel_tensor.bin $bit
	done
done


make clean
