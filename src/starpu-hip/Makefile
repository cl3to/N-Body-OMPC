# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2022-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

PROGS=nbody

include ./starpu.mk

nbody: nbody.o nbody_cpu.o

ifeq ($(USE_HIP),1)
nbody: nbody_hip.o
nbody: LDLIBS+=-L$(HIP_PATH)/lib -lamdhip64 -lstdc++
endif

compare:
	@DIFF_OUTPUT=$$( \
		diff ../debug/computed_pos_12 ../debug/solution_pos_12; \
		diff ../debug/computed_vel_12 ../debug/solution_vel_12); \
	if [ -z "$$DIFF_OUTPUT" ]; then \
		echo "All files match. No differences found."; \
	else \
		echo "The files differ. Note: hip and cpu funcitons may show different floating precision"; \
		echo "$$DIFF_OUTPUT"; \
	fi

compile:
	gcc -O2 -o bin2txt ../bin2txt.c

run-bin2txt: compile
	./bin2txt

compare-txt: run-bin2txt
	sdiff ../debug/computed_pos_12.txt ../debug/solution_pos_12.txt || true; \
	sdiff ../debug/computed_vel_12.txt ../debug/solution_vel_12.txt || true;
