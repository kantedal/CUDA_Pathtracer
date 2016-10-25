#ifndef RAYTRACER_CUDA_SPHERE_H
#define RAYTRACER_CUDA_SPHERE_H

#include <vector_types.h>
#include "cutil_math.h"
#include "Ray.h"
#include "Material.h"

struct Sphere {
    float3 position;
    float radius;
    int material_idx;

    Sphere(float3 _position, float _radius): position(_position), radius(_radius) {};
    Sphere(): position(make_float3(0,0,0)), radius(0) {};

    __device__ bool RayIntersection(Ray ray, float3 &collision_pos, float3 &collision_normal) {
		float3 op = position - ray.start_position;
		float t, epsilon = 0.000001f;
		float b = dot(op, ray.direction);
		float disc = b*b - dot(op, op) + radius * radius;
		if (disc<0) return false;
		else disc = sqrtf(disc);

		t = (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0);

        if (t < 0.1)
            return false;

		collision_pos = ray.start_position + ray.direction * t;
		float3 normal = normalize(collision_pos - position);
		collision_normal =  dot(normal, ray.direction) < 0 ? normal : normal * -1;

		return true;
    }
};


#endif