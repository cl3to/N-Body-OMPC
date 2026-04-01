#include <omp.h>

#include "../include/body.h"

#include <stdio.h>

void OMPC_NBody_Setup(const Pos *GlobalPos, const Vel *GlobalVel, const int *SendCounts,
                      const int *Displs, Pos **DevGlobalPos, Pos **DevLocalPos,
                      Vel **DevLocalVel, const int NumDevices, const int NBodies) {
    const int HostId = omp_get_initial_device();
    
    // This loop allocates the device buffers and copies initial data
    // The copy operations can be seen as the equivalent of MPI_Bcast and MPI_Scatterv
    // We use #pragma omp parallel for to overlap the operations on all devices
    // Each thread in this parallel region manages one device
    #pragma omp parallel for num_threads(NumDevices)
    for (int Device = 0; Device < NumDevices; ++Device) {
        // Allocate DevGlobalPos[Device], DevLocalPos[Device], DevLocalVel[Device]
        DevGlobalPos[Device] = omp_target_alloc(sizeof(Pos) * NBodies, Device);

        // The local_n for this device
        int LocalN = SendCounts[Device];

        DevLocalPos[Device] = omp_target_alloc(sizeof(Pos) * LocalN, Device);
        DevLocalVel[Device] = omp_target_alloc(sizeof(Vel) * LocalN, Device);

        // initially copy GlobalPos to DevGlobalPos[Device]
        // This is similar to MPI_Bcast
        omp_target_memcpy(DevGlobalPos[Device], GlobalPos,
                          sizeof(Pos) * NBodies, 0, 0, Device, HostId);

        // Copy the initial values ​​to LocalPos
        const Pos *SrcSlicePos = (DevGlobalPos[Device] + Displs[Device]);
        omp_target_memcpy(DevLocalPos[Device], SrcSlicePos,
                          sizeof(Pos) * LocalN, 0, 0, Device, Device);

        const Vel *SrcSlice = (GlobalVel + Displs[Device]);

        // Copy the initial slice of vel to DevLocalVel[Device]
        // This is similar to MPI_Scatterv
        omp_target_memcpy(DevLocalVel[Device],
                          SrcSlice,
                          sizeof(Vel) * LocalN, 0, 0, Device, HostId);
    }
}


void OMPC_Allgatherv(Pos *RootPtr, Pos **DevicePtrs, int *SendCounts,
                     const int *Displs, Pos **DeviceStaging,
                     const int NumDevices, const int NCount) {
    const int HostId = omp_get_initial_device();

    #pragma omp parallel for num_threads(NumDevices)
    for (int Device = 0; Device < NumDevices; ++Device) {
        int LocalN = SendCounts[Device];
        int LocalStart = Displs[Device];

        // Copy back the updated local positions from device to host
        omp_target_memcpy(&RootPtr[LocalStart],
                          DeviceStaging[Device],
                          sizeof(Pos) * LocalN, 0, 0, HostId, Device);
    }

    // Copy updated RootPtr back to all devices
    #pragma omp parallel for num_threads(NumDevices)
    for (int Device = 0; Device < NumDevices; ++Device) {
        omp_target_memcpy(DevicePtrs[Device], RootPtr,
                          sizeof(Pos) * NCount, 0, 0, Device, HostId);
    }    
}

void OMPC_Allgatherv_Ring(Pos *RootPtr, Pos **DevicePtrs, int *SendCounts,
                          const int *Displs, Pos **DeviceStaging,
                          const int NumDevices, const int NCount) {
    // Step 1: Copy local data to the global data of the same device
    #pragma omp parallel for num_threads(NumDevices)
    for (int Device = 0; Device < NumDevices; ++Device) {
        int LocalN = SendCounts[Device];
        int LocalStart = Displs[Device];
        Pos *DevLocalPtr  = (Pos *) DeviceStaging[Device];
        Pos *Dst = DevicePtrs[Device] + LocalStart;
    
        // Copy back the updated local positions from device to host
        omp_target_memcpy(Dst,
                          DevLocalPtr,
                          sizeof(Pos) * LocalN, 0, 0, Device, Device);
    }

    // Step 2: Iterate in the ring to propagate the data to all devices
    for (int s = 0; s < NumDevices; ++s) {
        #pragma omp parallel for num_threads(NumDevices)
        for (int Device = 0; Device < NumDevices; ++Device) {

            int DataIdx = (Device - s + NumDevices) % NumDevices;
            int NextDevice = (Device + 1) % NumDevices;

            int DataSize = SendCounts[DataIdx];
            int DataStart = Displs[DataIdx];

            Pos *SrcSlice = DevicePtrs[Device] + DataStart;
            Pos *DstSlice = DevicePtrs[NextDevice] + DataStart;

            // Copy back the updated local positions from device to host
            omp_target_memcpy(DstSlice,
                              SrcSlice, sizeof(Pos) * DataSize,
                              0, 0, NextDevice, Device);
        }
    }

}

void OMPC_Gatherv(Vel *RootPtr, Vel **DevicePtrs, int *SendCounts,
                     const int *Displs, Vel **DeviceStaging,
                     const int NumDevices, const int NCount) {
    const int HostId = omp_get_initial_device();

    #pragma omp parallel for num_threads(NumDevices)
    for (int Device = 0; Device < NumDevices; ++Device) {
        int LocalN = SendCounts[Device];
        int LocalStart = Displs[Device];

        // Copy back the updated LocalVel from devices to host
        omp_target_memcpy(&RootPtr[LocalStart],
                          DeviceStaging[Device],
                          sizeof(Vel) * LocalN, 0, 0, HostId, Device);
    }
}