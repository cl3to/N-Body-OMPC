STARPU_VERSION=1.4

# OpenMP target offload compiler (clang for ROCm, clang for NVIDIA)
OMP_CC ?= clang

# Override OMP_TARGET for NVIDIA GPUs: nvptx64-nvidia-cuda
OMP_TARGET ?= amdgcn-amd-amdhsa
GPU_ARCH ?= gfx90a
PARTITIONS_PER_RANK ?= 4

CFLAGS += -fopenmp -O3 -Wall -Wextra
CFLAGS += -fopenmp-targets=$(OMP_TARGET) -Xopenmp-target=$(OMP_TARGET) -march=$(GPU_ARCH)
CFLAGS += -DSTARPU_PARTITIONS_PER_RANK=$(PARTITIONS_PER_RANK)
# starpu.h pulls in hip_runtime.h via starpu_data_interfaces.h; define the AMD
# HIP platform so those headers parse even though we only use OpenMP target.
CFLAGS += -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1
ifeq ($(DEBUG),1)
CFLAGS += -DDEBUG
endif
LDLIBS += $(addprefix -L, $(subst :, ,$(LD_LIBRARY_PATH))) -lstarpu-$(STARPU_VERSION) -lstarpumpi-1.4 -lm

all: $(PROGS)

clean:
	rm -f $(PROGS) *.o */*.o */*/*.o
	rm -f paje.trace dag.dot *.rec trace.html starpu.log
	rm -f *.gp *.eps *.data
	rm -f bin2txt
