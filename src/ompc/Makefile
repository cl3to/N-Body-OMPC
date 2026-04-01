# Compiler and flags
CC = clang
CFLAGS = -fopenmp -fopenmp-targets=nvptx64 \
         -O3 -Wall -g
#          -Xopenmp-target=nvptx64 -march=sm_86
# x86_64-pc-linux-gnu
LDFLAGS = -lm
PYTHON ?= python3
RTOL ?= 1e-3
ATOL ?= 1e-5
MAX_REPORT ?= 10

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
	sdiff ../debug/computed_pos_12.txt ../debug/solution_pos_12.txt || true; \
	sdiff ../debug/computed_vel_12.txt ../debug/solution_vel_12.txt || true;
