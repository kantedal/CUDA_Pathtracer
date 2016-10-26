#ifndef RAYTRACER_CUDA_TRIANGLE_H
#define RAYTRACER_CUDA_TRIANGLE_H

#include <cstdio>
#include <iostream>
#include <vector>
#include "cutil_math.h"
#include "Ray.h"

struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
    float3 edge1;
    float3 edge2;
    float3 normal;
    int material_index;

    Triangle(float3 _v0, float3 _v1, float3 _v2): v0(_v0), v1(_v1), v2(_v2) {
        edge1 = v1 - v0;
        edge2 = v2 - v0;
        normal = normalize(cross(edge1, edge2));
    };
    Triangle(): v0(make_float3(0,0,0)), v1(make_float3(0,0,0)), v2(make_float3(0,0,0)) {};
};

#endif