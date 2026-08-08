# Compiler and flags
MPI_CC ?= clang
MPICC ?= MPICH_CC=$(MPI_CC) mpicc
HIPCC ?= hipcc
HIP_PATH ?= $(shell hipconfig -p 2>/dev/null || echo /opt/rocm)

BASE_CFLAGS = -fopenmp -O3 -Wall -std=c99
HIPCFLAGS = -O3 -Wall -std=c++11 -fPIC
LDFLAGS = -lm -L$(HIP_PATH)/lib -lamdhip64 -lstdc++ \
	-Wl,-rpath,$(HIP_PATH)/lib
PYTHON ?= python3
RTOL ?= 1e-3
ATOL ?= 1e-5
MAX_REPORT ?= 10

# Source files and target
SRCS = nbody.c nbody_cpu.c
HIP_SRCS = nbody_hip.cpp
OBJS = $(SRCS:.c=.o) $(HIP_SRCS:.hip=.o)
TARGET = nbody

# Default target
all: $(TARGET)

# Link object files to create the executable using MPI compiler
$(TARGET): $(OBJS)
	$(MPICC) $(BASE_CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile each .c file into .o using MPI compiler
%.o: %.c
	$(MPICC) $(BASE_CFLAGS) -c $< -o $@

# Compile the HIP kernels with hipcc
nbody_hip.o: nbody_hip.hip
	$(HIPCC) $(HIPCFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

compare:
	@$(PYTHON) ../compare.py \
		--computed ../debug/computed_pos_12 \
		--solution ../debug/solution_pos_12 \
		--kind pos \
		--rtol $(RTOL) --atol $(ATOL) --max-report $(MAX_REPORT)
	@$(PYTHON) ../compare.py \
		--computed ../debug/computed_vel_12 \
		--solution ../debug/solution_vel_12 \
		--kind vel \
		--rtol $(RTOL) --atol $(ATOL) --max-report $(MAX_REPORT)

compile:
	clang -O2 -o bin2txt ../bin2txt.c

run-bin2txt: compile
	./bin2txt

compare-txt: run-bin2txt
	@$(MAKE) compare

diff-txt: run-bin2txt
	sdiff ../debug/computed_pos_12.txt ../debug/solution_pos_12.txt || true; \
	sdiff ../debug/computed_vel_12.txt ../debug/solution_vel_12.txt || true;
