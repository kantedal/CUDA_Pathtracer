#ifndef RAYTRACER_CUDA_TRIANGLEC_H
#define RAYTRACER_CUDA_TRIANGLEC_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

texture<float4> triangle_texture;

__device__ bool TrianglesIntersection(Ray ray, int triangle_index, float3 &collision_pos, float3 &collision_normal, int &material_index) {
    float3 v0 = make_float3(tex1Dfetch(triangle_texture, triangle_index * 4));
    float3 edge1 = make_float3(tex1Dfetch(triangle_texture, triangle_index * 4 + 1));
    float3 edge2 = make_float3(tex1Dfetch(triangle_texture, triangle_index * 4 + 2));

    float EPS = 0.0001f;

    //Begin calculating determinant - also used to calculate u parameter
    float3 P = cross(ray.direction, edge2);
    float det = dot(edge1, P);

    if(det > -EPS && det < EPS) return false;
    float inv_det = 1.0f / det;

    //Distance from vertex1 to ray origin
    float3 T = make_float3(ray.start_position.x - v0.x, ray.start_position.y - v0.y, ray.start_position.z - v0.z);
    float u = dot(T, P);
    if (u < 0.0f || u > det) return false;

    float3 Q = cross(T, edge1);

    float v = dot(ray.direction, Q);
    if(v < 0.0f || u+v > det) return false;

    float t = dot(edge2, Q);

    if(t > EPS) {
        collision_pos = ray.start_position + inv_det * t * ray.direction;
        material_index = int(tex1Dfetch(triangle_texture, triangle_index * 4 + 3).x);
        collision_normal = normalize(cross(edge1, edge2));
        return true;
    }

    return false;
};

void bindTriangles(float *triangles_p, unsigned int triangle_count)
{
    triangle_texture.normalized = false;                      // access with normalized texture coordinates
    triangle_texture.filterMode = cudaFilterModePoint;        // Point mode, so no
    triangle_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

    size_t size = sizeof(float4) * triangle_count * 4;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaBindTexture(0, triangle_texture, triangles_p, channelDesc, size);
};

#endif