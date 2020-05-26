libdnn.so: dnn_avx.c
	gcc  -g -shared -fPIC -o $@ $^ -mavx2

clean:
	rm -f lib*.so
