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
    Pos *global_pos, Vel *local_vel, int global_start, int local_slice_start, int local_n, int n, int dev);
extern void integratePositions_cpu(Pos *local_pos, Vel *local_vel, int local_n);
extern void integratePositions_gpu(
    Pos *local_pos, Vel *local_vel, int local_slice_start, int local_n, int dev);

static void require_offload(int rank, int ndev) {
    int offload_ok = 0;
#pragma omp target device(ndev) map(tofrom : offload_ok)
    { offload_ok = !omp_is_initial_device(); }
    if (!offload_ok) {
        if (rank == 0) {
            fprintf(stderr,
                    "ERROR: OpenMP target offload not active. "
                    "Ensure libomptarget CUDA plugin is available and rebuild "
                    "with the correct GPU_ARCH (e.g. make GPU_ARCH=sm_75).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

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

    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        fprintf(stderr,
                "MPI rank %d can see no OpenMP target devices; a GPU is "
                "required.\n",
                rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int d = 0; d < num_devices; d++)
        require_offload(rank, d);

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

    omp_allocator_handle_t host_alloc = llvm_omp_target_host_mem_alloc;
    if (rank == 0) {
        global_pos = (Pos *)omp_alloc(sizeof(Pos) * nBodies, host_alloc);
        global_vel = (Vel *)omp_alloc(sizeof(Vel) * nBodies, host_alloc);
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

    Pos *local_pos = (Pos *)omp_alloc(sizeof(Pos) * local_n, host_alloc);
    Vel *local_vel = (Vel *)omp_alloc(sizeof(Vel) * local_n, host_alloc);

    MPI_Datatype MPI_Pos, MPI_Vel;
    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_Pos); // Assuming Pos is 3 floats
    MPI_Type_commit(&MPI_Pos);

    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_Vel); // Assuming Vel is 3 floats
    MPI_Type_commit(&MPI_Vel);

    if (rank != 0)
        global_pos = (Pos *)omp_alloc(sizeof(Pos) * nBodies, host_alloc);

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

#pragma omp parallel num_threads(num_devices)
    {
        int d = omp_get_thread_num();
        if (d >= num_devices)
            d = d % num_devices;
        int d_start = start_per_dev[d];
        int d_n = n_per_dev[d];
        int d_global_start = local_start + d_start;

#pragma omp target data device(d) \
        map(to : global_pos[0 : nBodies]) \
        map(tofrom : local_pos[d_start : d_n]) \
        map(tofrom : local_vel[d_start : d_n])
        {
            for (int iter = 0; iter < nIters; iter++) {
#if (BODYFORCE_USE_CPU == 1)
#pragma omp task
                bodyForce_cpu(global_pos,
                              local_vel + d_start,
                              d_global_start,
                              d_n,
                              nBodies);
#else
#pragma omp task
                bodyForce_gpu(global_pos,
                              local_vel,
                              d_global_start,
                              d_start,
                              d_n,
                              nBodies,
                              d);
#endif
#if (INTEGRATEPOSITIONS_USE_CPU == 1)
#pragma omp task
                integratePositions_cpu(local_pos + d_start,
                                       local_vel + d_start,
                                       d_n);
#else
#pragma omp task
                integratePositions_gpu(local_pos,
                                       local_vel,
                                       d_start,
                                       d_n,
                                       d);
#endif
#pragma omp taskwait
#pragma omp target update device(d) from(local_pos[d_start : d_n])
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
#pragma omp target update device(d) to(global_pos[0 : nBodies])
            }
        }
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

    omp_free(local_pos, host_alloc);
    omp_free(local_vel, host_alloc);
    omp_free(global_pos, host_alloc);
    if (rank == 0)
        omp_free(global_vel, host_alloc);
    free(sendCounts);
    free(displs);
    MPI_Finalize();
}
