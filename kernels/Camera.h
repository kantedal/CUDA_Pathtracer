#ifndef RAYTRACER_CUDA_CAMERA_H
#define RAYTRACER_CUDA_CAMERA_H

#include <vector_types.h>
#include <math.h>
#include "cutil_math.h"

struct Camera {
    float3 position;
    float3 look_at;
    float field_of_view;

    //Camera corner vertices
    float3 camera_v1;
    float3 camera_v2;
    float3 camera_v3;
    float3 camera_v4;

    Camera(float3 _position, float3 _look_at): position(_position), look_at(_look_at) {
        field_of_view = 3.14f / 4.0f;
        float d = 0.5f / tanf(field_of_view / 2.0f);
        camera_v1 = make_float3(0, -1/d, -1/d);
        camera_v2 = make_float3(0, 1/d, -1/d);
        camera_v3 = make_float3(0, 1/d, 1/d);
        camera_v4 = make_float3(0, -1/d, 1/d);
    };

    Camera() {
        field_of_view = 3.14f / 4.0f;
    };

    __device__ float3 GetRayDirection(float2 pixel_position) {
        float3 base_vector_x = camera_v3 - camera_v4;
        float3 base_vector_y = camera_v1 - camera_v4;
        float3 vert = camera_v4 + pixel_position.x / 512 * base_vector_x + (1.0f - pixel_position.y / 512) * base_vector_y;
        return normalize(vert - position);
    }
};
#endif