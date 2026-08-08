/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  University of Bordeaux, CNRS (LaBRI UMR 5800),
 * Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/*
 * This example demonstrates how to use StarPU to scale an array by a factor.
 * It shows how to manipulate data with StarPU's data management library.
 *  1- how to declare a piece of data to StarPU (starpu_vector_data_register)
 *  2- how to submit a task to StarPU
 *  3- how a kernel can manipulate the data (buffers[0].vector.ptr)
 */
#include <starpu.h>
#include <starpu_mpi.h>
#include <starpu_hip.h>

#include "../include/body.h"
#include "../include/files.h"

// #define DEBUG

#ifndef STARPU_PARTITIONS_PER_RANK
#define STARPU_PARTITIONS_PER_RANK 1
#endif

#if STARPU_PARTITIONS_PER_RANK < 1
#error "STARPU_PARTITIONS_PER_RANK must be at least 1"
#endif

extern void bodyForce_cpu(void *buffers[], void *_args);
extern void bodyForce_hip(void *buffers[], void *_args);
extern void integratePositions_cpu(void *buffers[], void *_args);
extern void integratePositions_hip(void *buffers[], void *_args);

static struct starpu_perfmodel bodyforce_perfmodel = {
    .type = STARPU_HISTORY_BASED, .symbol = "bodyforce"};

static struct starpu_perfmodel integratepositions_perfmodel = {
    .type = STARPU_HISTORY_BASED, .symbol = "integratepositions"};

static struct starpu_codelet bodyForce_cl = {
    .cpu_funcs = {bodyForce_cpu},

#ifdef STARPU_USE_HIP
    .hip_funcs = {bodyForce_hip},
    .hip_flags = {STARPU_HIP_ASYNC},
#endif
    .where = STARPU_HIP,
    .max_parallelism = INT_MAX,
    .nbuffers = 2,
    .modes = {STARPU_R, STARPU_RW},
    .model = &bodyforce_perfmodel,
};

static struct starpu_codelet integratePositions_cl = {
    .cpu_funcs = {integratePositions_cpu},

#ifdef STARPU_USE_HIP
    .hip_funcs = {integratePositions_hip},
    .hip_flags = {STARPU_HIP_ASYNC},
#endif
    .where = STARPU_HIP,
    .max_parallelism = INT_MAX,
    .nbuffers = 2,
    .modes = {STARPU_RW, STARPU_R},
    .model = &integratepositions_perfmodel,
};

int main(int argc, char **argv) {
    int rank, size, ret, nPartitions;
    int local_rank;
    int hip_devices;
    MPI_Comm local_comm;
    int nBodies = 2 << 12;
    Pos *pos;
    Vel *vel;
    starpu_mpi_tag_t tag = 0;

#ifndef DEBUG
    if (argc > 1)
        nBodies = 2 << (atoi(argv[1]) - 1);
#else
    printf("WARNING: Running on debug mode. Fixing nbodies to 2 << 12\n");
    (void)argc;
    (void)argv;
#endif

#ifdef DEBUG
    const char *initialized_pos = "../debug/initialized_pos_12";
    const char *initialized_vel = "../debug/initialized_vel_12";
    const char *computed_pos = "../debug/computed_pos_12";
    const char *computed_vel = "../debug/computed_vel_12";
#endif

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        0,
                        MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

    hipError_t hip_status = hipGetDeviceCount(&hip_devices);
    if (hip_status != hipSuccess || hip_devices == 0) {
        fprintf(stderr,
                "MPI rank %d: no HIP devices are available on this node\n",
                rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // One MPI process per node owns all visible GPUs. StarPU schedules codelets
    // across all of them, so we do not pin a single device per rank.
    if (local_rank != 0) {
        fprintf(stderr,
                "MPI rank %d: this implementation expects one MPI process per "
                "node (local_rank 0); local_rank is %d\n",
                rank,
                local_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    setbuf(stdout, NULL);
    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.sched_policy_name = "dmdar";
    conf.ncpus = 1;
    // Register every visible HIP device with StarPU so its scheduler can
    // distribute codelets across all node GPUs.
    conf.nhip = hip_devices;
    conf.use_explicit_workers_hip_gpuid = 0;

    starpu_mpi_init_conf(&argc, &argv, 0, MPI_COMM_WORLD, &conf);

    // Give StarPU enough tiles per rank to keep all GPUs busy. At least one
    // tile per GPU; allow more via STARPU_PARTITIONS_PER_RANK for pipelining.
    int partitions_per_rank = STARPU_PARTITIONS_PER_RANK;
    if (partitions_per_rank < hip_devices)
        partitions_per_rank = hip_devices;
    nPartitions = size * partitions_per_rank;
    if (nPartitions > nBodies) {
        fprintf(stderr,
                "MPI rank %d: %d partitions cannot be created for %d bodies\n",
                rank,
                nPartitions,
                nBodies);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        starpu_malloc_flags((void **)&pos,
                            sizeof(Pos) * nBodies,
                            STARPU_MALLOC_PINNED);
        starpu_malloc_flags((void **)&vel,
                            sizeof(Vel) * nBodies,
                            STARPU_MALLOC_PINNED);

#ifdef DEBUG
        read_values_from_file(initialized_pos, pos, sizeof(Pos), nBodies);
        read_values_from_file(initialized_vel, vel, sizeof(Vel), nBodies);
#else
        for (int i = 0; i < nBodies; i++) {
            pos[i].x = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
            pos[i].y = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
            pos[i].z = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
        }
        for (int i = 0; i < nBodies; i++) {
            vel[i].vx = ((float)rand() / (float)(RAND_MAX)) * 10.0f;
            vel[i].vy = ((float)rand() / (float)(RAND_MAX)) * 10.0f;
            vel[i].vz = ((float)rand() / (float)(RAND_MAX)) * 10.0f;
        }
#endif
    }

    // vectors are allocated in rank 0
    int memory_region = (rank == 0) ? STARPU_MAIN_RAM : -1;
    uintptr_t pos_ptr = (rank == 0) ? (uintptr_t)pos : 0;
    uintptr_t vel_ptr = (rank == 0) ? (uintptr_t)vel : 0;

    /* starpu data handles */
    starpu_data_handle_t pos_handle;
    starpu_vector_data_register(
        &pos_handle, memory_region, pos_ptr, nBodies, sizeof(Pos));
    starpu_data_handle_t *pos_handles = (starpu_data_handle_t *)malloc(
        sizeof(starpu_data_handle_t) * nPartitions);

    starpu_data_handle_t vel_handle;
    starpu_vector_data_register(
        &vel_handle, memory_region, vel_ptr, nBodies, sizeof(Vel));
    starpu_data_handle_t *vel_handles = (starpu_data_handle_t *)malloc(
        sizeof(starpu_data_handle_t) * nPartitions);

    // tagging mpi data
    starpu_mpi_data_register(pos_handle, tag++, 0);
    starpu_mpi_data_register(vel_handle, tag++, 0);

    // async partitioning vectors
    struct starpu_data_filter filter = {
        .filter_func = starpu_vector_filter_block, .nchildren = nPartitions};
    starpu_data_partition_plan(pos_handle, &filter, pos_handles);
    starpu_data_partition_plan(vel_handle, &filter, vel_handles);
    for (int i = 0; i < nPartitions; i++) {
        starpu_mpi_data_register(vel_handles[i], tag++, 0);
        starpu_mpi_data_register(pos_handles[i], tag++, 0);
    }

    const int nIters = 10;
    double start = starpu_timing_now();

    for (int i = 0; i < nIters; i++) {
        for (int j = 0; j < nPartitions; j++) {
            ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
                                         &bodyForce_cl,
                                         STARPU_R,
                                         pos_handle,
                                         STARPU_RW,
                                         vel_handles[j],
                                         STARPU_EXECUTE_ON_NODE,
                                         j % size,
                                         0);
        }

        for (int j = 0; j < nPartitions; j++) {
            ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
                                         &integratePositions_cl,
                                         STARPU_RW,
                                         pos_handles[j],
                                         STARPU_R,
                                         vel_handles[j],
                                         STARPU_EXECUTE_ON_NODE,
                                         j % size,
                                         0);
        }
    }
    starpu_task_wait_for_all();

    starpu_data_unpartition_submit(vel_handle, nPartitions, vel_handles, -1);
    starpu_data_unpartition_submit(pos_handle, nPartitions, pos_handles, -1);
    starpu_data_partition_clean(pos_handle, nPartitions, pos_handles);
    starpu_data_partition_clean(vel_handle, nPartitions, vel_handles);

    if (rank == 0) {
        starpu_data_acquire(pos_handle, STARPU_R);
        starpu_data_acquire(vel_handle, STARPU_R);
        pos = starpu_data_get_local_ptr(pos_handle);
        vel = starpu_data_get_local_ptr(vel_handle);
        double timing = (starpu_timing_now() - start) / 1.0e6;
        printf("runtime: %lf\n", timing); // seconds, matching src/openmp
#ifdef DEBUG
        write_values_to_file(computed_pos, pos, sizeof(Pos), nBodies);
        write_values_to_file(computed_vel, vel, sizeof(Vel), nBodies);
#endif
        starpu_data_release(pos_handle);
        starpu_data_release(vel_handle);
    }

    starpu_data_unregister(pos_handle);
    starpu_data_unregister(vel_handle);

    if (rank == 0) {
        starpu_free_flags(pos, sizeof(Pos) * nBodies, STARPU_MALLOC_PINNED);
        starpu_free_flags(vel, sizeof(Vel) * nBodies, STARPU_MALLOC_PINNED);
    }
    free(pos_handles);
    free(vel_handles);

    starpu_mpi_shutdown();
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
}
