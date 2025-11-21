#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/body.h"
#include "../include/files.h"

// #define DEBUG
#define BODYFORCE_USE_CPU 0
#define INTEGRATEPOSITIONS_USE_CPU 0

extern void bodyForce_cpu(
    Pos *GlobalPos, Vel *local_vel, int local_start, int local_n, int n);
extern void bodyForce_gpu(Pos *GlobalPos, Vel *local_vel, int local_start, int local_n, int n);
extern void integratePositions_cpu(Pos *local_pos, Vel *local_vel, int local_n);
extern void integratePositions_gpu(Pos *local_pos, Vel *local_vel, int local_n);


int main(int argc, char **argv) {
    int nBodies = 2 << 12;
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

    const int NumDevices = omp_get_num_devices();
    const int HostId = omp_get_initial_device();

    int base = nBodies / NumDevices;
    int rem = nBodies % NumDevices;
    int *sendCounts = malloc(NumDevices * sizeof(int));
    int *displs = malloc(NumDevices * sizeof(int));
    int offset = 0;
    for (int i = 0; i < NumDevices; i++) {
        int count = (i < rem) ? base + 1 : base;
        sendCounts[i] = count;
        displs[i] = offset;
        offset += sendCounts[i];
    }

    // Host-side global arrays
    Pos *GlobalPos = (Pos *)malloc(sizeof(Pos) * nBodies);
    Vel *GlobalVel = (Vel *)malloc(sizeof(Vel) * nBodies);

    // Initialize positions and velocities
#ifdef DEBUG
    read_values_from_file(
        initialized_pos, GlobalPos, sizeof(Pos), nBodies);
    read_values_from_file(
        initialized_vel, GlobalVel, sizeof(Vel), nBodies);
#else
    for (int i = 0; i < nBodies; ++i) {
        GlobalPos[i].x = ((float)rand() / (float)RAND_MAX) * 100.0f;
        GlobalPos[i].y = ((float)rand() / (float)RAND_MAX) * 100.0f;
        GlobalPos[i].z = ((float)rand() / (float)RAND_MAX) * 100.0f;
        GlobalVel[i].vx = ((float)rand() / (float)RAND_MAX) * 10.0f;
        GlobalVel[i].vy = ((float)rand() / (float)RAND_MAX) * 10.0f;
        GlobalVel[i].vz = ((float)rand() / (float)RAND_MAX) * 10.0f;
    }
#endif

    // Allocate local buffers on host to hold data for each device
    Pos **HostLocalPos = malloc(sizeof(Pos *) * NumDevices);
    Vel **HostLocalVel = malloc(sizeof(Vel *) * NumDevices);
    for (int Device = 0; Device < NumDevices; ++Device) {
        int ln = sendCounts[Device];
        HostLocalPos[Device] = (Pos *)malloc(sizeof(Pos) * ln);
        HostLocalVel[Device] = (Vel *)malloc(sizeof(Vel) * ln);
    }

    // Persistent allocation on devices (using OpenMP runtime functions)
    // each device will have:
    //   - DevGlobalPos[Device] = complete copy of GlobalPos on the device
    //   - DevLocalPos[Device], DevLocalVel[Device] = local buffers on the device
    void **DevGlobalPos = malloc(sizeof(void *) * NumDevices);
    void **DevLocalPos = malloc(sizeof(void *) * NumDevices);
    void **DevLocalVel = malloc(sizeof(void *) * NumDevices);

    // This loop allocates the device buffers and copies initial data
    // The copy operations can be seen as the equivalent of MPI_Bcast and MPI_Scatterv
    // We use #pragma omp parallel for to overlap the operations on all devices
    // Each thread in this parallel region manages one device
    #pragma omp parallel for num_threads(NumDevices)
    for (int Device = 0; Device < NumDevices; ++Device) {
        // Allocate DevGlobalPos[Device], DevLocalPos[Device], DevLocalVel[Device]
        DevGlobalPos[Device] = omp_target_alloc(sizeof(Pos) * nBodies, Device);

        // The local_n for this device
        int LocalN = sendCounts[Device];

        DevLocalPos[Device] = omp_target_alloc(sizeof(Pos) * LocalN, Device);
        DevLocalVel[Device] = omp_target_alloc(sizeof(Vel) * LocalN, Device);

        // initially copy GlobalPos to DevGlobalPos[Device]
        // This is similar to MPI_Bcast
        omp_target_memcpy(DevGlobalPos[Device], GlobalPos,
                          sizeof(Pos) * nBodies, 0, 0, Device, HostId);

        // Copy the initial slice of vel to DevLocalVel[Device]
        // This is similar to MPI_Scatterv
        omp_target_memcpy(DevLocalVel[Device],
                          &GlobalVel[displs[Device]],
                          sizeof(Vel) * LocalN, 0, 0, Device, HostId);
    }

    const int nIters = 10;
    start = omp_get_wtime();

    for(int Iter = 0; Iter < nIters; ++Iter) {
        // For each device, launch the bodyForce and integratePositions kernels
        #pragma omp parallel for num_threads(NumDevices)
        for (int Device = 0; Device < NumDevices; ++Device) {
            int LocalN = sendCounts[Device];
            int LocalStart = displs[Device];

            // Set the device for this thread
            omp_set_default_device(Device);

            // Launch bodyForce
            bodyForce_gpu((Pos *)DevGlobalPos[Device],
                          (Vel *)DevLocalVel[Device],
                          LocalStart,
                          LocalN,
                          nBodies);

            // Launch integratePositions
            integratePositions_gpu((Pos *)DevLocalPos[Device],
                                   (Vel *)DevLocalVel[Device],
                                   LocalN);
        }

        // After computation, gather results back to host
        // This is similar to MPI_Allgatherv
        #pragma omp parallel for num_threads(NumDevices)
        for (int Device = 0; Device < NumDevices; ++Device) {
            int LocalN = sendCounts[Device];

            // Copy back the updated local positions from device to host
            omp_target_memcpy(HostLocalPos[Device],
                              DevLocalPos[Device],
                              sizeof(Pos) * LocalN, 0, 0, HostId, Device);
        }

        // Now assemble GlobalPos from HostLocalPos
        for (int Device = 0; Device < NumDevices; ++Device) {
            int LocalN = sendCounts[Device];
            int LocalStart = displs[Device];
            for (int i = 0; i < LocalN; ++i) {
                GlobalPos[LocalStart + i] = HostLocalPos[Device][i];
            }
        }

        // Copy updated GlobalPos back to all devices
        #pragma omp parallel for num_threads(NumDevices)
        for (int Device = 0; Device < NumDevices; ++Device) {
            omp_target_memcpy(DevGlobalPos[Device], GlobalPos,
                              sizeof(Pos) * nBodies, 0, 0, Device, HostId);
        }
    }

    printf("%lf\n", omp_get_wtime() - start); // seconds

#ifdef DEBUG
    write_values_to_file(computed_pos, GlobalPos, sizeof(Pos), nBodies);
    write_values_to_file(computed_vel, GlobalVel, sizeof(Vel), nBodies);
#endif

    // Free memory
    for (int Device = 0; Device < NumDevices; ++Device) {
        omp_target_free(DevGlobalPos[Device], Device);
        omp_target_free(DevLocalPos[Device], Device);
        omp_target_free(DevLocalVel[Device], Device);
        free(HostLocalPos[Device]);
        free(HostLocalVel[Device]);
    }

    free(DevGlobalPos);
    free(DevLocalPos);
    free(DevLocalVel);
    free(HostLocalPos);
    free(HostLocalVel);
    free(GlobalPos);
    free(GlobalVel);
    free(sendCounts);
    free(displs);

    return 0;
}
