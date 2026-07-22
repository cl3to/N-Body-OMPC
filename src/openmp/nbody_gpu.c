#include "../include/body.h"
#include <math.h>
#include <omp.h>

static int gpu_device;

void set_gpu_device(int device) {
    gpu_device = device;
}

void bodyForce_gpu(
    Pos *global_pos, Vel *local_vel, int local_start, int local_n, int n) {
#pragma omp target teams distribute parallel for map(to : global_pos[0 : n])   \
    map(tofrom : local_vel[0 : local_n]) thread_limit(64) device(gpu_device)
    for (int i = 0; i < local_n; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;
        int global_i = local_start + i;
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

        local_vel[i].vx += dt * Fx;
        local_vel[i].vy += dt * Fy;
        local_vel[i].vz += dt * Fz;
    }
}

void integratePositions_gpu(Pos *local_pos, Vel *local_vel, int local_n) {
#pragma omp target teams distribute parallel for map(                          \
        tofrom : local_pos[0 : local_n]) map(to : local_vel[0 : local_n])      \
    thread_limit(64) device(gpu_device)
    for (int i = 0; i < local_n; i++) {
        local_pos[i].x += local_vel[i].vx * dt;
        local_pos[i].y += local_vel[i].vy * dt;
        local_pos[i].z += local_vel[i].vz * dt;
    }
}
