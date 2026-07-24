#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/body.h"
#include "../include/files.h"

// #define DEBUG

extern void bodyForce_gpu(Pos *GlobalPos, Vel *local_vel, int local_start, int local_n, int n, int device);
extern void integratePositions_gpu(Pos *local_pos, Vel *local_vel, int local_n, int device);

extern void OMPC_NBody_Setup(const Pos *GlobalPos, const Vel *GlobalVel, const int *SendCounts,
                             const int *Displs, Pos **DevGlobalPos, Pos **DevLocalPos,
                             Vel **DevLocalVel, const int NumDevices, const int NBodies);

extern void OMPC_Allgatherv(Pos *RootPtr, Pos **DevicePtrs, int *SendCounts,
                     const int *Displs, Pos **DeviceStaging,
                     const int NumDevices, const int NCount);

extern void OMPC_Allgatherv_Ring(Pos *RootPtr, Pos **DevicePtrs, int *SendCounts,
                          const int *Displs, Pos **DeviceStaging,
                          const int NumDevices, const int NCount);

extern void OMPC_Gatherv(Vel *RootPtr, Vel **DevicePtrs, int *SendCounts,
                         const int *Displs, Vel **DeviceStaging,
                         const int NumDevices, const int NCount);

                          
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

    // Persistent allocation on devices (using OpenMP runtime functions)
    // each device will have:
    //   - DevGlobalPos[Device] = complete copy of GlobalPos on the device
    //   - DevLocalPos[Device], DevLocalVel[Device] = local buffers on the device
    Pos **DevGlobalPos = malloc(sizeof(Pos *) * NumDevices);
    Pos **DevLocalPos = malloc(sizeof(Pos *) * NumDevices);
    Vel **DevLocalVel = malloc(sizeof(Vel *) * NumDevices);

    // This allocates the device buffers and copies initial data
    // The copy operations can be seen as the equivalent of MPI_Bcast and MPI_Scatterv
    OMPC_NBody_Setup(GlobalPos, GlobalVel, sendCounts, displs, DevGlobalPos,
                     DevLocalPos, DevLocalVel, NumDevices, nBodies);

    const int nIters = 10;
    start = omp_get_wtime();

    for(int Iter = 0; Iter < nIters; ++Iter) {
        // For each device, launch the bodyForce and integratePositions kernels
        #pragma omp parallel for num_threads(NumDevices)
        for (int Device = 0; Device < NumDevices; ++Device) {
            int LocalN = sendCounts[Device];
            int LocalStart = displs[Device];

            // Launch bodyForce
            bodyForce_gpu((Pos *)DevGlobalPos[Device],
                          (Vel *)DevLocalVel[Device],
                          LocalStart,
                          LocalN,
                          nBodies,
                          Device);

            // Launch integratePositions
            integratePositions_gpu((Pos *)DevLocalPos[Device],
                                   (Vel *)DevLocalVel[Device],
                                   LocalN,
                                   Device);
        }

        // After computation, gather results across devices
        // This is similar to MPI_Allgatherv
        OMPC_Allgatherv_Ring(GlobalPos, DevGlobalPos, sendCounts, displs,
                             DevLocalPos, NumDevices, nBodies);
    }

    // After computation, gather results back to host
    omp_target_memcpy(GlobalPos, DevGlobalPos[0], sizeof(Pos) * nBodies, 0, 0, HostId, 0);

    OMPC_Gatherv(GlobalVel, DevLocalVel,  sendCounts, displs,
                 DevLocalVel, NumDevices, nBodies);

    printf("runtime: %lf\n", omp_get_wtime() - start); // seconds

#ifdef DEBUG
    write_values_to_file(computed_pos, GlobalPos, sizeof(Pos), nBodies);
    write_values_to_file(computed_vel, GlobalVel, sizeof(Vel), nBodies);
#endif

    // Free memory
    for (int Device = 0; Device < NumDevices; ++Device) {
        omp_target_free(DevGlobalPos[Device], Device);
        omp_target_free(DevLocalPos[Device], Device);
        omp_target_free(DevLocalVel[Device], Device);
    }

    free(DevGlobalPos);
    free(DevLocalPos);
    free(DevLocalVel);
    free(GlobalPos);
    free(GlobalVel);
    free(sendCounts);
    free(displs);

    return 0;
}
