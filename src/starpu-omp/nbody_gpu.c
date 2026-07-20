#include <starpu.h>
#include <starpu_mpi.h>

#include "../include/body.h"

void bodyForce_gpu(void *buffers[], void *_args) {
    (void)_args;

    unsigned int nPos = STARPU_VECTOR_GET_NX(buffers[0]);
    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

    unsigned int offset = STARPU_VECTOR_GET_OFFSET(buffers[1]) / sizeof(Vel);

#pragma omp target teams distribute parallel for \
    map(to : p[0 : nPos]) map(tofrom : v[0 : nVel]) thread_limit(64)
    for (unsigned i = 0; i < nVel; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;
        unsigned global_i = offset + i;
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
    (void)_args;

    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

#pragma omp target teams distribute parallel for \
    map(tofrom : p[0 : nVel]) map(to : v[0 : nVel]) thread_limit(64)
    for (unsigned i = 0; i < nVel; i++) {
        p[i].x += v[i].vx * dt;
        p[i].y += v[i].vy * dt;
        p[i].z += v[i].vz * dt;
    }
}
