CXXFLAGS=-O3 -std=c++14
LDFLAGS=-lm
OBJECTS=main.o lib.o conv.o

convolution: main.cpp lib.cpp conv.cu
	nvcc $(CXXFLAGS) -o $@	$^ $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) *.o convolution
