#ifndef RAYTRACER_CUDA_RENDERKERNEL_H
#define RAYTRACER_CUDA_RENDERKERNEL_H

#include <cuda.h>
#include "Camera.h"
#include "Material.h"
#include "Sphere2.h"

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
);

#endif