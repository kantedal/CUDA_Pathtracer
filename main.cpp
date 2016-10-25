#include <iostream>
#include "Graphics.h"
#include "Scene.h"
#include "kernels/Triangle.h"

float3* accumulated_buffer;

Graphics graphics;
Scene scene;

Triangle* cpuTriangles;
Triangle* triangles;
int triangle_count;

Material* cpuMaterials;
Material* materials;
int material_count;

Sphere* cpuSpheres;
Sphere* spheres;
int sphere_count;


Camera* camera;

void launch_kernel_render(float3* pixels, float3* accumulated, Triangle* triangles, int triangle_count, Sphere* spheres, int sphere_count, Material* materials, int material_count, Camera* camera);

void RenderSample(float3* pixels) {
    launch_kernel_render(pixels, accumulated_buffer, triangles, triangle_count, spheres, sphere_count, materials, material_count, camera);
}

int main(int argc, char** argv) {
    scene = Scene();

    cpuMaterials = scene.GenerateMaterialArray(material_count);
    cudaMalloc((void**)&materials, material_count * sizeof(Material));
    cudaMemcpy(materials, cpuMaterials, material_count * sizeof(Material), cudaMemcpyHostToDevice);

    cpuTriangles = scene.GenerateTriangleArray(triangle_count);
    cudaMalloc((void**)&triangles, triangle_count * sizeof(Triangle));
    cudaMemcpy(triangles, cpuTriangles, triangle_count * sizeof(Triangle), cudaMemcpyHostToDevice);

    cpuSpheres = scene.GenerateSphereArray(sphere_count);
    cudaMalloc((void**)&spheres, sphere_count * sizeof(Sphere));
    cudaMemcpy(spheres, cpuSpheres, sphere_count * sizeof(Sphere), cudaMemcpyHostToDevice);

    cudaMalloc(&accumulated_buffer, window_width * window_height * sizeof(float3));

    camera = scene.get_camera();

    graphics = Graphics(&argc, argv);

    return 0;
}