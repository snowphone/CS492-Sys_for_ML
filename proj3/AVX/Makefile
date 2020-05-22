libdnn.so: dnn_avx.c
	gcc  -Wall -g -shared -fPIC -o $@ $^ -mavx2

dnn_avx: dnn_avx.c
	gcc  -Wall -g  -o $@ $^ -mavx2 -lm -lpthread

clean:
	rm -f lib*.so
