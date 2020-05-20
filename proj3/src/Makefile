matmul.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^

clean:
	rm -f matmul.so
