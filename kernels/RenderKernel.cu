//
// Created by Filip K on 24/10/16.
//

#ifndef RAYTRACER_CUDA_RENDERKERNEL_H
#define RAYTRACER_CUDA_RENDERKERNEL_H

#include <cuda.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <iostream>
#include "cutil_math.h"
#include "Triangle.h"
#include "Camera.h"
#include "Material.h"
#include "Sphere2.h"

int frame = 1;
float g_time = 0.0f;

union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

uint WangHash(uint a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

__device__ bool RayIntersection(
        Ray ray, Triangle* triangles, int triangle_count,
        Sphere* spheres, int sphere_count, Material* materials,
        float3 &collision_pos, float3 &collision_normal, Material &collision_material
) {

    for (int sphere_idx = 0; sphere_idx < sphere_count; sphere_idx++) {
        if (spheres[sphere_idx].RayIntersection(ray, collision_pos, collision_normal)) {
            //collision_normal = normalize(collision_pos - spheres[sphere_idx].position);
            collision_material = materials[4];
            //printf("%.2f %.2f %.2f \n", collision_pos.x, collision_pos.y, collision_pos.z);
            //printf("%.2f %.2f %.2f \n", collision_normal.x, collision_normal.y, collision_normal.z);
            return true;
        }
    }

    for (int tri_idx = 0; tri_idx < triangle_count; tri_idx++) {
        if (triangles[tri_idx].TriangleRayIntersection(ray, collision_pos)) {
            collision_normal = triangles[tri_idx].normal;
            collision_material = materials[triangles[tri_idx].material_idx];
            return true;
        }
    }
    return false;
}

__device__ float3 PathTrace(Ray ray, Triangle* triangles, int triangle_count, Sphere* spheres, int sphere_count, Material* materials, curandState *randState) {

    float3 mask = make_float3(1,1,1);
    float3 accumulated_color = make_float3(0,0,0);

    for (int iteration = 0; iteration < 5; iteration++) {
        float3 collision_normal = make_float3(0,0,0);
        float3 collision_pos = make_float3(0,0,0);
        float3 reflected_dir = make_float3(0,0,0);
        float3 transmitted_dir = make_float3(0,0,0);
        Material collision_material;

        // First chack so that ray intersects scene
        if (!RayIntersection(ray, triangles, triangle_count, spheres, sphere_count, materials, collision_pos, collision_normal, collision_material))
            return make_float3(0,0,0);

        mask *= collision_material.BRDF(ray.direction, collision_normal);

        if (collision_material.type == EMISSION) {
            accumulated_color += (mask * collision_material.color * collision_material.emission_rate);
            break;
        }

        collision_material.PDF(randState, ray.direction, collision_normal, reflected_dir, transmitted_dir);

        if (!isZero(reflected_dir)) {
            ray = Ray(collision_pos + reflected_dir * 0.01f, reflected_dir);
        }
        else {
            break;
        }
    }

    return accumulated_color;
}

__global__ void render_kernel(
        float3* d_pixels, float3* d_accumulated,
        Triangle* triangles, int triangle_count,
        Sphere* spheres, int sphere_count,
        Material* materials, int material_count,
        Camera camera, int frame, uint hash_frame,
        unsigned int width, unsigned int height
) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1) * width + x;

    if (frame == 1) {
        d_accumulated[i] = make_float3(0, 0, 0);
    }

    curandState randState;
    int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    curand_init(hash_frame + threadId, 0, 0, &randState);

    Ray ray = Ray(camera.position, camera.GetRayDirection(make_float2(x, y)));
    d_accumulated[i] += PathTrace(ray, triangles, triangle_count, spheres, sphere_count, materials, &randState);
    float3 temp_clr = d_accumulated[i] / frame;

    float3 clr = make_float3(clamp(temp_clr.x, 0.0f, 1.0f), clamp(temp_clr.y, 0.0f, 1.0f), clamp(temp_clr.z, 0.0f, 1.0f));

	Colour fcolour;
    fcolour.components = make_uchar4(
            (unsigned char)(powf(clr.x, 1 / 2.2f) * 255),
            (unsigned char)(powf(clr.y, 1 / 2.2f) * 255),
            (unsigned char)(powf(clr.z, 1 / 2.2f) * 255), 1);

    d_pixels[i] = make_float3(x, y, fcolour.c);
}

void launch_kernel_render(float3* pixels, float3* accumulated_buffer, Triangle* triangles, int triangle_count, Sphere* spheres, int sphere_count, Material* materials, int material_count, Camera* camera) {
    dim3 block(32, 32, 1);
    dim3 grid(512 / block.x, 512 / block.y, 1);
    render_kernel<<<grid, block>>>(pixels, accumulated_buffer, triangles, triangle_count, spheres, sphere_count, materials, material_count, *camera, frame, WangHash(frame), 512, 512);

    g_time += 0.01;
    frame++;

    if (frame % 10 == 0)
        std::cout << "Rendering iteration: " << frame << std::endl;
}

#endif