#include <starpu.h>
#include <starpu_mpi.h>
#include <starpu_worker.h>
#include <omp.h>

#include "../include/body.h"

void bodyForce_gpu(void *buffers[], void *_args) {
    int mpi_rank = 0;
    int global_start = 0;
    if (_args)
        starpu_codelet_unpack_args(_args, &mpi_rank, &global_start);
    (void)mpi_rank;

    // Retrieve pointers managed directly by StarPU's GPU memory manager
    Pos *dev_p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *dev_v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

    unsigned int nPos = STARPU_VECTOR_GET_NX(buffers[0]);
    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    // Run the OpenMP target region on the GPU bound to the StarPU HIP worker
    // executing this codelet, so it operates on the StarPU device pointers.
    int dev_id = starpu_worker_get_devid(starpu_worker_get_id());

    int num_threads = 64;
    int num_blocks = (nVel + num_threads - 1) / num_threads;

#pragma omp target teams distribute parallel for \
    is_device_ptr(dev_p, dev_v) device(dev_id) thread_limit(num_threads) num_teams(num_blocks)
    for (unsigned i = 0; i < nVel; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;
        unsigned global_i = global_start + i;
        for (unsigned j = 0; j < nPos; j++) {
            float dx = dev_p[j].x - dev_p[global_i].x;
            float dy = dev_p[j].y - dev_p[global_i].y;
            float dz = dev_p[j].z - dev_p[global_i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = my_rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        dev_v[i].vx += dt * Fx;
        dev_v[i].vy += dt * Fy;
        dev_v[i].vz += dt * Fz;
    }
}

void integratePositions_gpu(void *buffers[], void *_args) {
    int mpi_rank = 0;
    if (_args)
        starpu_codelet_unpack_args(_args, &mpi_rank);
    (void)mpi_rank;

    Pos *dev_p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *dev_v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    // Run the OpenMP target region on the GPU bound to the StarPU HIP worker
    // executing this codelet, so it operates on the StarPU device pointers.
    int dev_id = starpu_worker_get_devid(starpu_worker_get_id());

#pragma omp target teams distribute parallel for \
    is_device_ptr(dev_p, dev_v) device(dev_id) thread_limit(64)
    for (unsigned i = 0; i < nVel; i++) {
        dev_p[i].x += dev_v[i].vx * dt;
        dev_p[i].y += dev_v[i].vy * dt;
        dev_p[i].z += dev_v[i].vz * dt;
    }
}
