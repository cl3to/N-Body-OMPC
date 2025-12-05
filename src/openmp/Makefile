# Compiler and flags
export MPICH_CC=clang
MPICC = mpicc
CFLAGS = -fopenmp -fopenmp-targets=nvptx64 \
         -O3 -Wall -std=c99
LDFLAGS = -lm

# -Xopenmp-target=nvptx64 -march=sm_80 \

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
	@DIFF_OUTPUT=$$( \
		diff ../debug/computed_pos_12 ../debug/solution_pos_12; \
		diff ../debug/computed_vel_12 ../debug/solution_vel_12); \
	if [ -z "$$DIFF_OUTPUT" ]; then \
		echo "All files match. No differences found."; \
	else \
		echo "The files differ. Note: cuda and cpu functions may show different floating precision"; \
		echo "$$DIFF_OUTPUT"; \
	fi

compile:
	clang -O2 -o bin2txt ../bin2txt.c

run-bin2txt: compile
	./bin2txt

compare-txt: run-bin2txt
	sdiff ../debug/computed_pos_12.txt ../debug/solution_pos_12.txt || true; \
	sdiff ../debug/computed_vel_12.txt ../debug/solution_vel_12.txt || true;
