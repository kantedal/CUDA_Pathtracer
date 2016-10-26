//
// Created by Filip K on 24/10/16.
//

#include <iostream>
#include "Scene.h"

Scene::Scene() {
    CreateDefaultScene();
}

void Scene::CreateDefaultScene() {
    Material* diffuse_red_material = new Material(DIFFUSE, make_float3(1,0,0));
    Material* diffuse_green_material = new Material(DIFFUSE, make_float3(0,1,0));
    Material* diffuse_blue_material = new Material(DIFFUSE, make_float3(0,0,1));
    Material* diffuse_white_material = new Material(DIFFUSE, make_float3(1,1,1));
    Material* specular_white_material = new Material(SPECULAR, make_float3(1,1,1));
    Material* transmission_white_material = new Material(TRANSMISSION, make_float3(1,1,1));
    Material* emission_material = new Material(EMISSION, make_float3(0.6,0.8,1));
    emission_material->emission_rate = 40.0f;

    materials.push_back(diffuse_red_material);
    materials.push_back(diffuse_green_material);
    materials.push_back(diffuse_blue_material);
    materials.push_back(diffuse_white_material);
    materials.push_back(specular_white_material);
    materials.push_back(transmission_white_material);
    materials.push_back(emission_material);

    Object3d light_plane = Object3d::loadObj("../models/light_plane.obj", emission_material);
    Object3d floor = Object3d::loadObj("../models/floor.obj", diffuse_white_material);
    Object3d roof = Object3d::loadObj("../models/roof.obj", diffuse_white_material);
    Object3d left_wall = Object3d::loadObj("../models/left_wall.obj", diffuse_green_material);
    Object3d right_wall = Object3d::loadObj("../models/right_wall.obj", diffuse_red_material);

    objects.push_back(light_plane);
    objects.push_back(floor);
    objects.push_back(roof);
    objects.push_back(left_wall);
    objects.push_back(right_wall);

    Sphere sphere1 = Sphere(make_float3(5, -3.5, -2.5f), 2.0f);
    Sphere sphere2 = Sphere(make_float3(8, 1.5, -2.5f), 2.0f);

    spheres.push_back(sphere1);
    spheres.push_back(sphere2);

    camera = new Camera(make_float3(-1,0,0), make_float3(1,0,0));
}

Material* Scene::GenerateMaterialArray(int &material_size) {
    material_size = (int) materials.size();
    Material* new_materials = new Material[material_size];

    for (unsigned int mat_idx = 0; mat_idx < material_size; mat_idx++) {
        new_materials[mat_idx] = *materials.at(mat_idx);
    }

    return new_materials;
}

Triangle* Scene::GenerateTriangleArray(int &triangle_size) {
    std::vector<Triangle> allTris;

    triangle_size = 0;
    for (unsigned int obj_idx = 0; obj_idx < objects.size(); obj_idx++) {

        int material_index = 0;
        for (int mat_idx = 0; mat_idx < materials.size(); mat_idx++) {
            if (materials.at(mat_idx) == objects.at(obj_idx).get_material()) {
                material_index = mat_idx;
                break;
            }
        }

        for (unsigned int tri_idx = 0; tri_idx < objects.at(obj_idx).get_triangles().size(); tri_idx++) {
            float3 v0 = make_float3(objects.at(obj_idx).get_triangles().at(tri_idx).v0.x, objects.at(obj_idx).get_triangles().at(tri_idx).v0.y, objects.at(obj_idx).get_triangles().at(tri_idx).v0.z);
            float3 v1 = make_float3(objects.at(obj_idx).get_triangles().at(tri_idx).v1.x, objects.at(obj_idx).get_triangles().at(tri_idx).v1.y, objects.at(obj_idx).get_triangles().at(tri_idx).v1.z);
            float3 v2 = make_float3(objects.at(obj_idx).get_triangles().at(tri_idx).v2.x, objects.at(obj_idx).get_triangles().at(tri_idx).v2.y, objects.at(obj_idx).get_triangles().at(tri_idx).v2.z);

            Triangle new_tri = Triangle(v0, v1, v2);
            new_tri.material_index = material_index;

            allTris.push_back(new_tri);
            triangle_size++;
        }
    }

    Triangle* triangles = new Triangle[triangle_size];

    for (int i = 0; i < triangle_size; i++) {
        triangles[i] = allTris.at(i);
    }

    return triangles;
}

Sphere* Scene::GenerateSphereArray(int &sphere_count) {
    sphere_count = (int) spheres.size();
    Sphere* new_spheres = new Sphere[sphere_count];

    for (unsigned int sphere_idx = 0; sphere_idx < sphere_count; sphere_idx++) {
        new_spheres[sphere_idx].material_idx = 1;
        new_spheres[sphere_idx] = spheres.at(sphere_idx);
    }

    return new_spheres;
}

float* Scene::GenerateTriangles(int &triangle_count) {
    float *triangles_p;
    std::vector<float4> all_triangles;

    for (unsigned int obj_idx = 0; obj_idx < objects.size(); obj_idx++) {
        int material_index = 0;
        for (int mat_idx = 0; mat_idx < materials.size(); mat_idx++) {
            if (materials.at(mat_idx) == objects.at(obj_idx).get_material()) {
                material_index = mat_idx;
                break;
            }
        }

        for (unsigned int tri_idx = 0; tri_idx < objects.at(obj_idx).get_triangles().size(); tri_idx++) {
            Triangle triangle = objects.at(obj_idx).get_triangles().at(tri_idx);
            float4 v0 = make_float4(triangle.v0.x, triangle.v0.y, triangle.v0.z, 0);
            float4 edge1 = make_float4(triangle.v1.x - triangle.v0.x, triangle.v1.y - triangle.v0.y, triangle.v1.z - triangle.v0.z, 0);
            float4 edge2 = make_float4(triangle.v2.x - triangle.v0.x, triangle.v2.y - triangle.v0.y, triangle.v2.z - triangle.v0.z, 0);
            float4 material = make_float4(material_index, 0,0,0);

            all_triangles.push_back(v0);
            all_triangles.push_back(edge1);
            all_triangles.push_back(edge2);
            all_triangles.push_back(material);
        }
    }

    size_t triangle_size = all_triangles.size() * sizeof(float4);
    triangle_count = (int) all_triangles.size() / 4;

    if (triangle_size > 0)
    {
        cudaMalloc((void **)&triangles_p, triangle_size);
        cudaMemcpy(triangles_p, &all_triangles[0], triangle_size, cudaMemcpyHostToDevice);
    }

    return triangles_p;
}


