//
// Created by Filip K on 24/10/16.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include "RenderKernel.h"
#include "Triangle.cuh"

int frame = 1;
float g_time = 0.0f;
clock_t begin_time;

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
        Ray ray, int triangle_count,
        Sphere* spheres, int sphere_count,
        float3 &collision_pos, float3 &collision_normal, int &material_index
) {

    for (int sphere_idx = 0; sphere_idx < sphere_count; sphere_idx++) {
        if (spheres[sphere_idx].RayIntersection(ray, collision_pos, collision_normal)) {
            material_index = 5;
            return true;
        }
    }

    for (int tri_idx = 0; tri_idx < triangle_count; tri_idx++) {
        if (TrianglesIntersection(ray, tri_idx, collision_pos, collision_normal, material_index)) {
            return true;
        }
    }

    return false;

}

__device__ float3 PathTrace(Ray ray, int triangle_count, Sphere* spheres, int sphere_count, Material* materials, curandState *randState) {
    float3 mask = make_float3(1,1,1);
    float3 accumulated_color = make_float3(0,0,0);

    for (int iteration = 0; iteration < 5; iteration++) {
        float3 collision_normal, collision_pos, next_dir = make_float3(0,0,0);
        float distribution = 1.0f;
        int material_index = 0;


        if (!RayIntersection(ray, triangle_count, spheres, sphere_count, collision_pos, collision_normal, material_index))
            return make_float3(0,0,0);

        Material collision_material = materials[material_index];
        mask *= collision_material.BRDF(ray.direction, collision_normal) * distribution;

        if (collision_material.type == EMISSION) {
            accumulated_color += (mask * collision_material.color * collision_material.emission_rate);
            break;
        }

        collision_material.PDF(randState, ray.direction, collision_normal, next_dir, distribution);

        if (!isZero(next_dir)) {
            ray = Ray(collision_pos + next_dir * 0.01f, next_dir);
        }
        else {
            break;
        }
    }

    return accumulated_color;
}

__global__ void render_kernel(
        float3* d_pixels, float3* d_clr_pixels, float3* d_accumulated,
        int triangle_count,
        Sphere* spheres, int sphere_count,
        Material* materials,
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

    const int samples = 8;
    float3 new_clr = make_float3(0,0,0);
    float3* positions = camera.GetRayDirections2(make_float2(x,y), samples, &randState);

    for (int sample = 0; sample < samples; sample++) {
        Ray ray = Ray(camera.position, positions[sample]);
        new_clr += PathTrace(ray, triangle_count, spheres, sphere_count, materials, &randState);
    }
    new_clr /= (float) samples;

    d_accumulated[i] += new_clr;
    float3 temp_clr = d_accumulated[i] / frame;

    float3 clr = make_float3(clamp(temp_clr.x, 0.0f, 1.0f), clamp(temp_clr.y, 0.0f, 1.0f), clamp(temp_clr.z, 0.0f, 1.0f));

    Colour fcolour;
    fcolour.components = make_uchar4(
            (unsigned char)(powf(clr.x, 1 / 2.2f) * 255),
            (unsigned char)(powf(clr.y, 1 / 2.2f) * 255),
            (unsigned char)(powf(clr.z, 1 / 2.2f) * 255), 1);

    d_pixels[i] = make_float3(x, y, fcolour.c);
    d_clr_pixels[i] = make_float3(clr.x, clr.y, clr.z);
}

void launch_kernel_render(
        float3* pixels,
        float3* clrPixels,
        float3* accumulated_buffer,
        float* triangles_f,
        int triangle_count_f,
        Sphere* spheres,
        int sphere_count,
        Material* materials,
        int material_count,
        Camera* camera
) {
    if (frame % 50 == 0)
        clock_t begin_time = clock();

    if (frame == 1) {
        bindTriangles(triangles_f, triangle_count_f);
    }

    dim3 block(32, 32, 1);
    dim3 grid(512 / block.x, 512 / block.y, 1);
    render_kernel<<<grid, block>>>(pixels, clrPixels, accumulated_buffer, triangle_count_f, spheres, sphere_count, materials, *camera, frame, WangHash(frame), 512, 512);

    g_time += 0.01;
    frame++;

    if (frame % 50 == 0) {
        std::cout << "Rendering iteration: " << frame << ", time elapsed: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    }

}
