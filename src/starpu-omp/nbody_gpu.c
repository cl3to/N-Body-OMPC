#include <starpu.h>
#include <starpu_mpi.h>
#include <omp.h>

#include "../include/body.h"

void bodyForce_gpu(void *buffers[], void *_args) {
    int mpi_rank = 0;
    int global_start = 0;
    if (_args)
        starpu_codelet_unpack_args(_args, &mpi_rank, &global_start);
    int num_devices = omp_get_num_devices();
    int dev_id = num_devices > 0 ? (mpi_rank % num_devices) : 0;

    unsigned int nPos = STARPU_VECTOR_GET_NX(buffers[0]);
    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

#pragma omp target teams distribute parallel for \
    map(to : p[0 : nPos]) map(tofrom : v[0 : nVel]) thread_limit(64) \
    device(dev_id)
    for (unsigned i = 0; i < nVel; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;
        unsigned global_i = global_start + i;
        for (unsigned j = 0; j < nPos; j++) {
            float dx = p[j].x - p[global_i].x;
            float dy = p[j].y - p[global_i].y;
            float dz = p[j].z - p[global_i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = my_rsqrtf(distSqr);
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

void integratePositions_gpu(void *buffers[], void *_args) {
    int mpi_rank = 0;
    if (_args)
        starpu_codelet_unpack_args(_args, &mpi_rank);
    int num_devices = omp_get_num_devices();
    int dev_id = num_devices > 0 ? (mpi_rank % num_devices) : 0;

    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

#pragma omp target teams distribute parallel for \
    map(tofrom : p[0 : nVel]) map(to : v[0 : nVel]) thread_limit(64) \
    device(dev_id)
    for (unsigned i = 0; i < nVel; i++) {
        p[i].x += v[i].vx * dt;
        p[i].y += v[i].vy * dt;
        p[i].z += v[i].vz * dt;
    }
}
