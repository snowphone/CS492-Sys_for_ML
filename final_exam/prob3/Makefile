CXXFLAGS=-g -O3 -Wall -std=c++14 -mavx2
LDFLAGS=-lm -lpthread
OBJECTS=main.o lib.o

convolution: $(OBJECTS)
	$(CXX) -o $@	$^ $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) *.o convolution
