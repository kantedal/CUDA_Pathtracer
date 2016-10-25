#ifndef RAYTRACER_CUDA_RAY_H
#define RAYTRACER_CUDA_RAY_H

struct Ray {
    float3 start_position;
    float3 direction;

    __device__ Ray(float3 _start_position, float3 _direction)
        :start_position(_start_position), direction(_direction) {};

    __device__ Ray() {};
};

#endif
