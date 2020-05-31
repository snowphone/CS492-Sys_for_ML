libcudadnn.so: dnn_cuda.cu
	nvcc -std=c++11  -g -Xcompiler -shared -Xcompiler -fPIC -o $@ $^ -Xcompiler -mavx2

clean:
	rm -f lib*.so
