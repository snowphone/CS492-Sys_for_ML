cublas.so: dnn_cublas.cu
	nvcc -shared -Xcompiler -fPIC -o $@ $^ -lcublas -lopenblas

clean:
	rm -f cublas.so
