# Compiler and flags
MPI_CC ?= clang
MPICC ?= MPICH_CC=$(MPI_CC) mpicc
OMP_TARGET ?= nvptx64
GPU_ARCH ?= sm_80

BASE_CFLAGS = -fopenmp -O3 -Wall -std=c99
OFFLOAD_CFLAGS_CLANG = -fopenmp-targets=$(OMP_TARGET) \
	-Xopenmp-target=$(OMP_TARGET) -march=$(GPU_ARCH)
LDFLAGS = -lm
PYTHON ?= python3
RTOL ?= 1e-3
ATOL ?= 1e-5
MAX_REPORT ?= 10

CFLAGS = $(BASE_CFLAGS) $(OFFLOAD_CFLAGS_CLANG)

# Source files and target
SRCS = nbody.c nbody_cpu.c nbody_gpu.c
OBJS = $(SRCS:.c=.o)
TARGET = nbody

# Default target
all: $(TARGET)

# Link object files to create the executable using MPI compiler
$(TARGET): $(OBJS)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile each .c file into .o using MPI compiler
%.o: %.c
	$(MPICC) $(CFLAGS) -c $< -o $@

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
