#ifndef RAYTRACER_CUDA_TRIANGLE_H
#define RAYTRACER_CUDA_TRIANGLE_H

#include <vector_types.h>
#include "cutil_math.h"
#include "Ray.h"

struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
    float3 normal;
    int material_idx;

    Triangle(float3 _v0, float3 _v1, float3 _v2): v0(_v0), v1(_v1), v2(_v2) {
        CalculateNormal();
    };
    Triangle(): v0(make_float3(0,0,0)), v1(make_float3(0,0,0)), v2(make_float3(0,0,0)) {};


    void CalculateNormal() {
        float3 baseVec1 = v1 - v0;
        float3 baseVec2 = v2 - v0;
        normal = normalize(cross(baseVec1, baseVec2));
    }

    __device__ bool TriangleRayIntersection(Ray ray, float3 &collision_pos) {
        float EPS = 0.0001;

        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;

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
            return true;
        }

        return false;
    }
};


#endif