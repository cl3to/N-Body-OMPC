STARPU_VERSION=1.4

# OpenMP target offload compiler (amdclang for ROCm, clang for NVIDIA)
OMP_CC ?= amdclang

# Override OMP_TARGET for NVIDIA GPUs: nvptx64-nvidia-cuda
OMP_TARGET ?= amdgcn-amd-amdhsa
GPU_ARCH ?= gfx90a
PARTITIONS_PER_RANK ?= 4

CFLAGS += -fopenmp -O3 -Wall -Wextra
CFLAGS += -fopenmp-targets=$(OMP_TARGET) -Xopenmp-target=$(OMP_TARGET) -march=$(GPU_ARCH)
CFLAGS += -DSTARPU_PARTITIONS_PER_RANK=$(PARTITIONS_PER_RANK)
LDLIBS += $(addprefix -L, $(subst :, ,$(LD_LIBRARY_PATH))) -lstarpu-$(STARPU_VERSION) -lstarpumpi-1.4 -lm

all: $(PROGS)

clean:
	rm -f $(PROGS) *.o */*.o */*/*.o
	rm -f paje.trace dag.dot *.rec trace.html starpu.log
	rm -f *.gp *.eps *.data
	rm -f bin2txt
