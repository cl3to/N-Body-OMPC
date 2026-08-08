#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/body.h"
#include "../include/files.h"

// #define DEBUG
#define BODYFORCE_USE_CPU 0
#define INTEGRATEPOSITIONS_USE_CPU 0

extern void bodyForce_cpu(
    Pos *global_pos, Vel *local_vel, int local_start, int local_n, int n);
extern void bodyForce_gpu(
    Pos *dev_global_pos, Vel *dev_local_vel, int global_start, int local_n, int n, int dev);
extern void integratePositions_cpu(Pos *local_pos, Vel *local_vel, int local_n);
extern void integratePositions_gpu(
    Pos *dev_local_pos, Vel *dev_local_vel, int local_n, int dev);

extern int hip_get_num_devices(void);
extern void hip_init_device(int dev);
extern void *hip_alloc_host(size_t bytes);
extern void hip_free_host(void *ptr);
extern void *hip_alloc_dev(size_t bytes);
extern void hip_free_dev(void *ptr);
extern void hip_copy_host_to_dev(
    void *dst, const void *src, size_t bytes, int dev);
extern void hip_copy_dev_to_host(
    void *dst, const void *src, size_t bytes, int dev);
extern void hip_device_sync(int dev);

int main(int argc, char **argv) {
    int nBodies = 2 << 12;
    int rank, size;
    double start;

#ifdef DEBUG
    const char *initialized_pos = "../debug/initialized_pos_12";
    const char *initialized_vel = "../debug/initialized_vel_12";
    const char *computed_pos = "../debug/computed_pos_12";
    const char *computed_vel = "../debug/computed_vel_12";
#endif

#ifndef DEBUG
    if (argc > 1)
        nBodies = 2 << (atoi(argv[1]) - 1);
#else
    (void)argc;
    (void)argv;
    printf("WARNING: Running on debug mode. Fixing nbodies to 2 << 12\n");
#endif

    // MPI initialization. MPI is called from within an OpenMP single region
    // (MPI_Allgatherv between iterations), so request thread support.
    int mpi_thread_provided;
    MPI_Init_thread(
        &argc, &argv, MPI_THREAD_SERIALIZED, &mpi_thread_provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (mpi_thread_provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr,
                "MPI rank %d: MPI does not support MPI_THREAD_SERIALIZED "
                "(provided %d)\n",
                rank,
                mpi_thread_provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int num_devices = hip_get_num_devices();
    if (num_devices == 0) {
        fprintf(stderr,
                "MPI rank %d can see no HIP devices; a GPU is required.\n",
                rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int d = 0; d < num_devices; d++)
        hip_init_device(d);

    int base = nBodies / size;
    int rem = nBodies % size;
    int *sendCounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int count = (i < rem) ? base + 1 : base;
        sendCounts[i] = count;
        displs[i] = offset;
        offset += sendCounts[i];
    }

    int local_n = sendCounts[rank];
    int local_start = displs[rank];

    // Partition this rank's local bodies across all visible GPUs (round-robin
    // contiguous chunks so each GPU owns a contiguous slice of local_n).
    int *n_per_dev = malloc(num_devices * sizeof(int));
    int *start_per_dev = malloc(num_devices * sizeof(int));
    int bbase = local_n / num_devices;
    int brem = local_n % num_devices;
    int soffset = 0;
    for (int d = 0; d < num_devices; d++) {
        n_per_dev[d] = bbase + (d < brem ? 1 : 0);
        start_per_dev[d] = soffset;
        soffset += n_per_dev[d];
    }

    Pos *global_pos = NULL;
    Vel *global_vel = NULL;

    // Allocate pinned host memory so every device can copy into these buffers
    // asynchronously (hipMemcpyAsync requires host memory to be registered).
    if (rank == 0) {
        global_pos = (Pos *)hip_alloc_host(sizeof(Pos) * nBodies);
        global_vel = (Vel *)hip_alloc_host(sizeof(Vel) * nBodies);
#ifdef DEBUG
        read_values_from_file(
            initialized_pos, global_pos, sizeof(Pos), nBodies);
        read_values_from_file(
            initialized_vel, global_vel, sizeof(Vel), nBodies);
#else
        for (int i = 0; i < nBodies; i++) {
            global_pos[i].x = ((float)rand() / (float)RAND_MAX) * 100.0f;
            global_pos[i].y = ((float)rand() / (float)RAND_MAX) * 100.0f;
            global_pos[i].z = ((float)rand() / (float)RAND_MAX) * 100.0f;
            global_vel[i].vx = ((float)rand() / (float)RAND_MAX) * 10.0f;
            global_vel[i].vy = ((float)rand() / (float)RAND_MAX) * 10.0f;
            global_vel[i].vz = ((float)rand() / (float)RAND_MAX) * 10.0f;
        }
#endif
    }

    Pos *local_pos = (Pos *)hip_alloc_host(sizeof(Pos) * local_n);
    Vel *local_vel = (Vel *)hip_alloc_host(sizeof(Vel) * local_n);

    MPI_Datatype MPI_Pos, MPI_Vel;
    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_Pos); // Assuming Pos is 3 floats
    MPI_Type_commit(&MPI_Pos);

    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_Vel); // Assuming Vel is 3 floats
    MPI_Type_commit(&MPI_Vel);

    if (rank != 0)
        global_pos = (Pos *)hip_alloc_host(sizeof(Pos) * nBodies);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        start = omp_get_wtime();
    }

    MPI_Scatterv(global_vel,
                 sendCounts,
                 displs,
                 MPI_Vel,
                 local_vel,
                 local_n,
                 MPI_Vel,
                 0,
                 MPI_COMM_WORLD);
    MPI_Bcast(global_pos, nBodies, MPI_Pos, 0, MPI_COMM_WORLD);
    memcpy(local_pos, global_pos + local_start, sizeof(Pos) * local_n);

    const int nIters = 10;

    // Use N OpenMP threads: one per GPU device. Each thread pins a HIP device,
    // keeps a persistent device copy of the full global_pos plus its own slice
    // of local_pos/local_vel, then runs all nIters iterations. Per iteration:
    // launch force + integrate kernels on the device stream, flush the
    // local_pos slice to the host, hit a barrier so all devices finish, then a
    // single thread performs MPI_Allgatherv on the host, then each thread
    // pushes the updated global_pos back to its device. This keeps data
    // resident on each GPU for all 10 iterations while synchronizing global_pos
    // across MPI ranks each iter.
#pragma omp parallel num_threads(num_devices)
    {
        int d = omp_get_thread_num();
        if (d >= num_devices)
            d = d % num_devices;
        int d_start = start_per_dev[d];
        int d_n = n_per_dev[d];
        int d_global_start = local_start + d_start;

        hip_init_device(d);

        Pos *dev_global_pos = (Pos *)hip_alloc_dev(sizeof(Pos) * nBodies);
        Pos *dev_local_pos = (Pos *)hip_alloc_dev(sizeof(Pos) * d_n);
        Vel *dev_local_vel = (Vel *)hip_alloc_dev(sizeof(Vel) * d_n);

        hip_copy_host_to_dev(
            dev_global_pos, global_pos, sizeof(Pos) * nBodies, d);
        hip_copy_host_to_dev(
            dev_local_vel, local_vel + d_start, sizeof(Vel) * d_n, d);
        hip_copy_host_to_dev(
            dev_local_pos, local_pos + d_start, sizeof(Pos) * d_n, d);
        hip_device_sync(d);

        for (int iter = 0; iter < nIters; iter++) {
#if (BODYFORCE_USE_CPU == 1)
            bodyForce_cpu(global_pos,
                          local_vel + d_start,
                          d_global_start,
                          d_n,
                          nBodies);
#else
            bodyForce_gpu(dev_global_pos,
                          dev_local_vel,
                          d_global_start,
                          d_n,
                          nBodies,
                          d);
#endif
#if (INTEGRATEPOSITIONS_USE_CPU == 1)
            integratePositions_cpu(local_pos + d_start,
                                   local_vel + d_start,
                                   d_n);
#else
            integratePositions_gpu(dev_local_pos, dev_local_vel, d_n, d);
#endif
#if (BODYFORCE_USE_CPU == 0 || INTEGRATEPOSITIONS_USE_CPU == 0)
            hip_copy_dev_to_host(
                local_pos + d_start, dev_local_pos, sizeof(Pos) * d_n, d);
            hip_device_sync(d);
#endif
#pragma omp barrier
#pragma omp single
            {
                MPI_Allgatherv(local_pos,
                               local_n,
                               MPI_Pos,
                               global_pos,
                               sendCounts,
                               displs,
                               MPI_Pos,
                               MPI_COMM_WORLD);
            }
#if (BODYFORCE_USE_CPU == 0)
            // Same stream as bodyForce_gpu, so ordered before next iteration.
            hip_copy_host_to_dev(
                dev_global_pos, global_pos, sizeof(Pos) * nBodies, d);
#endif
        }

#if (INTEGRATEPOSITIONS_USE_CPU == 0)
        hip_copy_dev_to_host(
            local_vel + d_start, dev_local_vel, sizeof(Vel) * d_n, d);
        hip_device_sync(d);
#endif

        hip_free_dev(dev_global_pos);
        hip_free_dev(dev_local_pos);
        hip_free_dev(dev_local_vel);
    }
    MPI_Gatherv(local_vel,
                local_n,
                MPI_Vel,
                (rank == 0) ? global_vel : NULL,
                sendCounts,
                displs,
                MPI_Vel,
                0,
                MPI_COMM_WORLD);

    if (rank == 0)
        printf("runtime: %lf\n", omp_get_wtime() - start); // seconds

#ifdef DEBUG
    if (rank == 0) {
        write_values_to_file(computed_pos, global_pos, sizeof(Pos), nBodies);
        write_values_to_file(computed_vel, global_vel, sizeof(Vel), nBodies);
    }
#endif

    hip_free_host(local_pos);
    hip_free_host(local_vel);
    hip_free_host(global_pos);
    if (rank == 0)
        hip_free_host(global_vel);
    free(sendCounts);
    free(displs);
    free(n_per_dev);
    free(start_per_dev);
    MPI_Finalize();
}
