#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/body.h"
#include "../include/files.h"

#define DEBUG
#define BODYFORCE_USE_CPU 0
#define INTEGRATEPOSITIONS_USE_CPU 0

extern void bodyForce_cpu(
    Pos *global_pos, Vel *local_vel, int local_start, int local_n, int n);
extern void bodyForce_gpu(Pos *global_pos, Vel *local_vel, int local_start, int local_n, int n);
extern void integratePositions_cpu(Pos *local_pos, Vel *local_vel, int local_n);
extern void integratePositions_gpu(Pos *local_pos, Vel *local_vel, int local_n);

static void require_offload(int rank) {
    int offload_ok = 0;
#pragma omp target map(tofrom : offload_ok)
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

void bodyForce_mpi(
    Pos *global_pos, Vel *local_vel, int local_start, int local_n, int n) {
#if (BODYFORCE_USE_CPU == 1)
#pragma omp task
    bodyForce_cpu(global_pos, local_vel, local_start, local_n, n);
#else
#pragma omp task
    bodyForce_gpu(global_pos, local_vel, local_start, local_n, n);
#endif
}

void integratePositions_mpi(Pos *local_pos, Vel *local_vel, int local_n) {
#if (INTEGRATEPOSITIONS_USE_CPU == 1)
#pragma omp task
    integratePositions_cpu(local_pos, local_vel, local_n);
#else
#pragma omp task
    integratePositions_gpu(local_pos, local_vel, local_n);
#endif
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

    // MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    require_offload(rank);

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

    Pos *global_pos = NULL;
    Vel *global_vel = NULL;

    if (rank == 0) {
        global_pos = (Pos *)(malloc(sizeof(Pos) * nBodies));
        global_vel = (Vel *)(malloc(sizeof(Vel) * nBodies));
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

    Pos *local_pos = malloc(sizeof(Pos) * local_n);
    Vel *local_vel = malloc(sizeof(Vel) * local_n);

    MPI_Datatype MPI_Pos, MPI_Vel;
    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_Pos); // Assuming Pos is 3 floats
    MPI_Type_commit(&MPI_Pos);

    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_Vel); // Assuming Vel is 3 floats
    MPI_Type_commit(&MPI_Vel);

    if (rank != 0)
        global_pos = malloc(sizeof(Pos) * nBodies);

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


    for (int iter = 0; iter < nIters; iter++) {
        bodyForce_mpi(global_pos, local_vel, local_start, local_n, nBodies);
        integratePositions_mpi(local_pos, local_vel, local_n);
        MPI_Allgatherv(local_pos,
                       local_n,
                       MPI_Pos,
                       global_pos,
                       sendCounts,
                       displs,
                       MPI_Pos,
                       MPI_COMM_WORLD);
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
        printf("%lf\n", omp_get_wtime() - start); // seconds


#ifdef DEBUG
    if (rank == 0) {
        write_values_to_file(computed_pos, global_pos, sizeof(Pos), nBodies);
        write_values_to_file(computed_vel, global_vel, sizeof(Vel), nBodies);
    }
#endif

    free(local_pos);
    free(local_vel);
    free(global_pos);
    if (rank == 0)
        free(global_vel);
    free(sendCounts);
    free(displs);
    MPI_Finalize();
}
