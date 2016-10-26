#include <iostream>
#include "Graphics.h"
#include "Scene.h"
#include "kernels/RenderKernel.h"

float3* accumulated_buffer;

Graphics graphics;
Scene scene;

Triangle* cpuTriangles;
Triangle* triangles;
int triangle_count;
float* triangles_f;
int triangle_count_f;

Material* cpuMaterials;
Material* materials;
int material_count;

Sphere* cpuSpheres;
Sphere* spheres;
int sphere_count;

float4* firstCollision;

Camera* camera;


void RenderSample(float3* pixels, float3* clrPixels) {
    launch_kernel_render(pixels, clrPixels, accumulated_buffer, triangles_f, triangle_count_f, spheres, sphere_count, materials, material_count, camera);
}

int main(int argc, char** argv) {
    scene = Scene();

    // Allocate materials
    cpuMaterials = scene.GenerateMaterialArray(material_count);
    cudaMalloc((void**)&materials, material_count * sizeof(Material));
    cudaMemcpy(materials, cpuMaterials, material_count * sizeof(Material), cudaMemcpyHostToDevice);

    // Allocate spheres
    cpuSpheres = scene.GenerateSphereArray(sphere_count);
    cudaMalloc((void**)&spheres, sphere_count * sizeof(Sphere));
    cudaMemcpy(spheres, cpuSpheres, sphere_count * sizeof(Sphere), cudaMemcpyHostToDevice);

    triangles_f = scene.GenerateTriangles(triangle_count_f);

    // Allocate accumulated buffer
    cudaMalloc(&accumulated_buffer, window_width * window_height * sizeof(float3));


    camera = scene.get_camera();
    graphics = Graphics(&argc, argv);

    return 0;
}