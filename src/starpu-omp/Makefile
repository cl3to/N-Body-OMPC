PROGS=nbody

include ./starpu.mk

SRCS = nbody.c nbody_cpu.c nbody_gpu.c
OBJS = $(SRCS:.c=.o)

nbody: $(OBJS)
	MPICH_CC=$(OMP_CC) mpicc $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	MPICH_CC=$(OMP_CC) mpicc $(CFLAGS) $(CPPFLAGS) -c $< -o $@

compare:
	@DIFF_OUTPUT=$$( \
		diff ../debug/computed_pos_12 ../debug/solution_pos_12; \
		diff ../debug/computed_vel_12 ../debug/solution_vel_12); \
	if [ -z "$$DIFF_OUTPUT" ]; then \
		echo "All files match. No differences found."; \
	else \
		echo "The files differ. Note: cpu and gpu functions may show different floating precision"; \
		echo "$$DIFF_OUTPUT"; \
	fi

compile:
	gcc -O2 -o bin2txt ../bin2txt.c

run-bin2txt: compile
	./bin2txt

compare-txt: run-bin2txt
	sdiff ../debug/computed_pos_12.txt ../debug/solution_pos_12.txt || true; \
	sdiff ../debug/computed_vel_12.txt ../debug/solution_vel_12.txt || true;
