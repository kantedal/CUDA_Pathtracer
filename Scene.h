//
// Created by Filip K on 24/10/16.
//

#ifndef RAYTRACER_CUDA_SCENE_H
#define RAYTRACER_CUDA_SCENE_H

#include <vector>
#include "kernels/Material.h"
#include "Object3d.h"
#include "kernels/Camera.h"
#include "kernels/Sphere2.h"

class Scene {
public:
    Scene();

    Camera* get_camera() { return camera; }

    Triangle* GenerateTriangleArray(int &triangle_count);
    Material* GenerateMaterialArray(int &material_count);
    Sphere* GenerateSphereArray(int &sphere_count);
private:
    Camera* camera;
    std::vector<Object3d> objects;
    std::vector<Sphere> spheres;
    std::vector<Material*> materials;

    void CreateDefaultScene();
};


#endif //RAYTRACER_CUDA_SCENE_H
