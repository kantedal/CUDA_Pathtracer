//
// Created by Filip K on 24/10/16.
//

#ifndef RAYTRACER_CUDA_OBJECT3D_H
#define RAYTRACER_CUDA_OBJECT3D_H


#include <vector>
#include <string>
#include "kernels/Triangle.h"
#include "kernels/Material.h"

class Object3d {
public:
    Object3d(std::vector<Triangle> _triangles, Material* _material);

    std::vector<Triangle> get_triangles() { return triangles; }
    Material* get_material() { return material; }

    static Object3d loadObj(std::string filename, Material* _material);
private:
    Material* material;
    std::vector<Triangle> triangles;
};


#endif //RAYTRACER_CUDA_OBJECT3D_H
