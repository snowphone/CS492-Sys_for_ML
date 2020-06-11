CFLAGS=-g -O3
LDFLAGS=-lm
OBJECTS=main.o conv.o lib.o

convolution: $(OBJECTS)
	$(CC) -o $@	$^ $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) *.o convolution
