#ifndef RAYTRACER_CUDA_MATERIAL_H
#define RAYTRACER_CUDA_MATERIAL_H

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cutil_math.h"

#define DIFFUSE 0
#define SPECULAR 1
#define TRANSMISSION 2
#define EMISSION 3

__device__ void DiffusePDF(curandState *randstate, float3 direction, float3 collision_normal, float3 &reflected_dir);

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
                return color * dot(incoming_direction, -collision_normal) / M_PI;
            case (SPECULAR):
                return color;
            case (TRANSMISSION):
                return color;
            case (EMISSION):
                return color;
        }

        return make_float3(0,0,0);
    }

    __device__ void PDF(curandState *randstate, float3 direction, float3 collision_normal, float3 &next_dir, float &distribution) {
        if (type == DIFFUSE) {
            DiffusePDF(randstate, direction, collision_normal, next_dir);
        }
        else if (type == SPECULAR) {
            SpecularPDF(direction, collision_normal, next_dir);
        }
        else if (type == TRANSMISSION) {
            float3 real_normal = dot(collision_normal, direction) < 0 ? collision_normal : collision_normal * -1;
            TransmissionPDF(randstate, collision_normal, real_normal, direction, next_dir, distribution);
        }

    }

    // Diffuse pdf function
    __device__ void DiffusePDF(curandState *randstate, float3 direction, float3 collision_normal, float3 &next_dir) {
        float rand = curand_uniform(randstate);
        if (rand < 0.9) {
            float r1 = 2.0f *  ((float) M_PI) * curand_uniform(randstate);
            float r2 = curand_uniform(randstate);
            float r2s = sqrtf(r2);

            float3 w = collision_normal;
            float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w, u);

            // compute cosine weighted random ray direction on hemisphere
            next_dir = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));
        }
    }

    // Specular pdf function
    __device__ void SpecularPDF(float3 direction, float3 collision_normal, float3 &reflected_dir) {
        reflected_dir = normalize(direction - 2.0f * dot(direction, collision_normal) * collision_normal);
    }

    // Transmissive pdf function
    __device__ void TransmissionPDF(curandState *randstate, float3 collision_normal, float3 real_normal, float3 direction, float3 &next_dir, float &distribution) {
        bool into = dot(collision_normal, real_normal) > 0; // is ray entering or leaving refractive material?
        float nc = 1.0f;  // Index of Refraction air
        float nt = 1.3f;  // Index of Refraction glass/water
        float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
        float ddn = dot(direction, real_normal);
        float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

        if (cos2t < 0.0f) // total internal reflection
        {
            next_dir = normalize(direction - collision_normal * 2.0f * dot(collision_normal, direction));
        }
        else // cos2t > 0
        {
            // compute direction of transmission ray
            float3 tdir = direction * nnt;
            tdir -= normalize(collision_normal * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))));

            float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
            float c = 1.f - (into ? -ddn : dot(tdir, collision_normal));
            float Re = R0 + (1.f - R0) * c * c * c * c * c;
            float Tr = 1 - Re; // Transmission
            float P = 0.25f + 0.5f * Re;
            float RP = Re / P;
            float TP = Tr / (1.f - P);

            // randomly choose reflection or transmission ray
            if (curand_uniform(randstate) < 0.2) // reflection ray
            {
                distribution = RP;
                next_dir = normalize(direction - collision_normal * 2.0f * dot(collision_normal, direction));
            }
            else // transmission ray
            {
                distribution = TP;
                next_dir = normalize(tdir);
            }
        }
    }
};

#endif