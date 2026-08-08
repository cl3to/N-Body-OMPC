#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#include "../include/body.h"

#define MAX_DEVICES 32

static hipStream_t streams[MAX_DEVICES];
static int stream_created[MAX_DEVICES] = {0};

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err_ = (call);                                              \
        if (err_ != hipSuccess) {                                              \
            fprintf(stderr,                                                    \
                    "HIP error at %s:%d: %s (%s)\n",                           \
                    __FILE__,                                                  \
                    __LINE__,                                                  \
                    hipGetErrorName(err_),                                      \
                    hipGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static __global__ void
bodyForce(Pos *p, Vel *v, int nPos, int nVel, int offset) {
    int initialIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = initialIndex; i < nVel; i += stride) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < nPos; j++) {
            float dx = p[j].x - p[i + offset].x;
            float dy = p[j].y - p[i + offset].y;
            float dz = p[j].z - p[i + offset].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        v[i].vx += dt * Fx;
        v[i].vy += dt * Fy;
        v[i].vz += dt * Fz;
    }
}

static __global__ void integratePositions(Pos *p, Vel *v, int n) {
    int initialIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = initialIndex; i < n; i += stride) { // integrate position
        p[i].x += v[i].vx * dt;
        p[i].y += v[i].vy * dt;
        p[i].z += v[i].vz * dt;
    }
}

extern "C" int hip_get_num_devices(void) {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    return count;
}

extern "C" void hip_init_device(int dev) {
    if (dev >= MAX_DEVICES) {
        fprintf(stderr, "HIP error: device %d exceeds MAX_DEVICES\n", dev);
        exit(EXIT_FAILURE);
    }
    HIP_CHECK(hipSetDevice(dev));
    if (!stream_created[dev]) {
        HIP_CHECK(hipStreamCreate(&streams[dev]));
        stream_created[dev] = 1;
    }
}

extern "C" void *hip_alloc_host(size_t bytes) {
    void *ptr = NULL;
    HIP_CHECK(hipHostMalloc(&ptr, bytes, hipHostMallocPortable));
    return ptr;
}

extern "C" void hip_free_host(void *ptr) {
    HIP_CHECK(hipHostFree(ptr));
}

extern "C" void *hip_alloc_dev(size_t bytes) {
    void *ptr = NULL;
    HIP_CHECK(hipMalloc(&ptr, bytes));
    return ptr;
}

extern "C" void hip_free_dev(void *ptr) {
    HIP_CHECK(hipFree(ptr));
}

extern "C" void hip_copy_host_to_dev(
    void *dst, const void *src, size_t bytes, int dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(
        hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, streams[dev]));
}

extern "C" void hip_copy_dev_to_host(
    void *dst, const void *src, size_t bytes, int dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(
        hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, streams[dev]));
}

extern "C" void hip_device_sync(int dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(streams[dev]));
}

extern "C" void bodyForce_gpu(Pos *dev_global_pos,
                              Vel *dev_local_vel,
                              int global_start,
                              int local_n,
                              int n,
                              int dev) {
    unsigned threads_per_block = 64;
    unsigned nblocks = (local_n + threads_per_block - 1) / threads_per_block;

    bodyForce<<<nblocks, threads_per_block, 0, streams[dev]>>>(
        dev_global_pos, dev_local_vel, n, local_n, global_start);

    HIP_CHECK(hipGetLastError());
}

extern "C" void integratePositions_gpu(
    Pos *dev_local_pos, Vel *dev_local_vel, int local_n, int dev) {
    unsigned threads_per_block = 64;
    unsigned nblocks = (local_n + threads_per_block - 1) / threads_per_block;

    integratePositions<<<nblocks, threads_per_block, 0, streams[dev]>>>(
        dev_local_pos, dev_local_vel, local_n);

    HIP_CHECK(hipGetLastError());
}
