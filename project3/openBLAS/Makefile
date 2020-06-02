openblas.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^ -lopenblas

clean:
	rm -f openblas.so
