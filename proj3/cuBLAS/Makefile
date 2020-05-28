cublas.so: dnn_cublas.cu
	nvcc -shared -Xcompiler -fPIC -o $@ $^ -lcublas

clean:
	rm -f cublas.so
