#include "../include/body.h"
#include <math.h>
#include <omp.h>

void bodyForce_gpu(
    Pos *global_pos, Vel *local_vel, int global_start, int local_slice_start, int local_n, int n, int dev) {

    int num_threads = 64;
    int num_blocks = (local_n + num_threads - 1) / num_threads;

#pragma omp target teams distribute parallel for                               \
    map(present, alloc : global_pos[0 : n])                                    \
    map(present, alloc : local_vel[local_slice_start : local_n])               \
    num_teams(num_blocks) thread_limit(num_threads) device(dev)
    for (int i = 0; i < local_n; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;
        int global_i = global_start + i;
        for (unsigned j = 0; j < n; j++) {
            float dx = global_pos[j].x - global_pos[global_i].x;
            float dy = global_pos[j].y - global_pos[global_i].y;
            float dz = global_pos[j].z - global_pos[global_i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = my_rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        local_vel[local_slice_start + i].vx += dt * Fx;
        local_vel[local_slice_start + i].vy += dt * Fy;
        local_vel[local_slice_start + i].vz += dt * Fz;
    }
}

void integratePositions_gpu(Pos *local_pos, Vel *local_vel, int local_slice_start, int local_n, int dev) {
#pragma omp target teams distribute parallel for                              \
        map(present, alloc : local_pos[local_slice_start : local_n])          \
        map(present, alloc : local_vel[local_slice_start : local_n])          \
    thread_limit(64) device(dev)
    for (int i = 0; i < local_n; i++) {
        local_pos[local_slice_start + i].x += local_vel[local_slice_start + i].vx * dt;
        local_pos[local_slice_start + i].y += local_vel[local_slice_start + i].vy * dt;
        local_pos[local_slice_start + i].z += local_vel[local_slice_start + i].vz * dt;
    }
}
