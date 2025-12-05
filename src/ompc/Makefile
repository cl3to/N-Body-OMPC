# Compiler and flags
CC = clang
CFLAGS = -fopenmp -fopenmp-targets=nvptx64 \
         -O3 -Wall -g
#          -Xopenmp-target=nvptx64 -march=sm_86
# x86_64-pc-linux-gnu
LDFLAGS = -lm

# Source files and target (ajuste conforme seus arquivos finais)
SRCS = nbody.c nbody_gpu.c ompc_comm.c
OBJS = $(SRCS:.c=.o)
TARGET = nbody

# Default target
all: $(TARGET)

# Link object files
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile each .c file
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

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
