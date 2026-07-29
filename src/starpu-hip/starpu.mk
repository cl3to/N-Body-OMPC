STARPU_VERSION=1.4

CC = mpicc
HIPCC = hipcc

CPPFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION) --cflags starpumpi-1.4)
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION) --libs starpumpi-1.4)

CFLAGS += -O3 -Wall -Wextra -lm -fopenmp
HIPCCFLAGS = $(shell pkg-config --cflags starpu-$(STARPU_VERSION) --cflags starpumpi-1.4) -std=c++11 -fPIC

# HIP runtime root, used to locate libamdhip64.so. May be overridden by the
# environment (e.g. /opt/rocm).
HIP_PATH ?= $(shell hipconfig -p 2>/dev/null || echo /opt/rocm)

# to avoid having to use LD_LIBRARY_PATH
LDLIBS += -fopenmp -lm -Wl,-rpath -Wl,$(shell pkg-config --variable=libdir starpu-$(STARPU_VERSION))

# Automatically enable HIP
STARPU_CONFIG=$(shell pkg-config --variable=includedir starpu-$(STARPU_VERSION))/starpu/$(STARPU_VERSION)/starpu_config.h
ifneq ($(shell grep "STARPU_USE_HIP 1" $(STARPU_CONFIG)),)
USE_HIP=1
endif

%.o: %.hip
	$(HIPCC) $(HIPCCFLAGS) $< -c -o $@

all: $(PROGS)

clean:
	rm -f $(PROGS) *.o */*.o */*/*.o
	rm -f paje.trace dag.dot *.rec trace.html starpu.log
	rm -f *.gp *.eps *.data
	rm -f bin2txt
