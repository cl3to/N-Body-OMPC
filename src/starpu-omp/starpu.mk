STARPU_VERSION=1.4

CC = mpicc

CPPFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION) --cflags starpumpi-1.4)
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION) --libs starpumpi-1.4)

CFLAGS += -O3 -Wall -Wextra -lm -fopenmp

# to avoid having to use LD_LIBRARY_PATH
LDLIBS += -fopenmp -lm -Wl,-rpath -Wl,$(shell pkg-config --variable=libdir starpu-$(STARPU_VERSION))

all: $(PROGS)

clean:
	rm -f $(PROGS) *.o */*.o */*/*.o
	rm -f paje.trace dag.dot *.rec trace.html starpu.log
	rm -f *.gp *.eps *.data
	rm -f bin2txt
