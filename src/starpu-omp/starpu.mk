STARPU_VERSION=1.4

MPI_CC ?= clang
MPICC ?= MPICH_CC=$(MPI_CC) mpicc

OMP_TARGET ?= nvptx64
GPU_ARCH ?= sm_80

CPPFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION) --cflags starpumpi-1.4)
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION) --libs starpumpi-1.4)

CFLAGS += -fopenmp -O3 -Wall -Wextra -std=c99
CFLAGS += -fopenmp-targets=$(OMP_TARGET) -Xopenmp-target=$(OMP_TARGET) -march=$(GPU_ARCH)

LDLIBS += -lm -Wl,-rpath -Wl,$(shell pkg-config --variable=libdir starpu-$(STARPU_VERSION))

all: $(PROGS)

clean:
	rm -f $(PROGS) *.o */*.o */*/*.o
	rm -f paje.trace dag.dot *.rec trace.html starpu.log
	rm -f *.gp *.eps *.data
	rm -f bin2txt
