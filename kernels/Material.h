#ifndef RAYTRACER_CUDA_MATERIAL_H
#define RAYTRACER_CUDA_MATERIAL_H

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cutil_math.h"

#define DIFFUSE 0
#define SPECULAR 1
#define EMISSION 2

struct Material {
    float3 color;
    float emission_rate;
    int type;

    Material(int _type, float3 _color): type(_type), color(_color) {
        if (type == EMISSION) emission_rate = 1000.0f;
    };
    __device__ Material(): type(0), color(make_float3(0,0,0)), emission_rate(0) {};

    __device__ float3 BRDF(float3 incoming_direction, float3 collision_normal) {
        switch (type) {
            case (DIFFUSE):
                return color * 1.0f / 3.14f;// * dot(incoming_direction, -collision_normal);
            case (SPECULAR):
                return color;
            case (EMISSION):
                return color;
        }

        return make_float3(0,0,0);
    }

    __device__ void PDF(curandState *randstate, float3 direction, float3 collision_normal, float3 &reflected_dir, float3 &transmitted_dir) {
        if (type == DIFFUSE) {
            float rand = curand_uniform(randstate);
            if (rand < 0.8 && type != EMISSION) {
                float r1 = 2.0f *  M_PI * curand_uniform(randstate);
                float r2 = curand_uniform(randstate);
                float r2s = sqrtf(r2);

                float3 w = collision_normal;
                float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
                float3 v = cross(w, u);

                // compute cosine weighted random ray direction on hemisphere
                reflected_dir = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));
            }
        }
        else if (type == SPECULAR) {
            reflected_dir = normalize(direction - 2 * dot(direction, collision_normal) * collision_normal);
        }

    }
};


__device__

#endif