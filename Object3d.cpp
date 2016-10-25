//
// Created by Filip K on 24/10/16.
//

#include <fstream>
#include <sstream>
#include <iostream>
#include "Object3d.h"

Object3d::Object3d(std::vector<Triangle> _triangles, Material *_material): triangles(_triangles), material(_material) {}

Object3d Object3d::loadObj(std::string filename, Material* material) {
    std::vector<Triangle> triangles;
    std::vector<float3> vertices;

    std::string line;
    std::ifstream obj_file(filename);
    if (obj_file.is_open())
    {
        while (getline(obj_file, line))
        {
            std::istringstream iss(line);
            std::string type;
            iss >> type;

            if (type == "f") {
                std::string string_v0;
                std::string string_v1;
                std::string string_v2;
                iss >> string_v0 >> string_v1 >> string_v2;

                Triangle tri = Triangle(vertices.at(std::stoi(string_v0)-1), vertices.at(std::stoi(string_v1)-1), vertices.at(std::stoi(string_v2)-1));
                triangles.push_back(tri);
            }
            else if (type == "v") {
                std::string string_x;
                std::string string_y;
                std::string string_z;
                iss >> string_x >> string_y >> string_z;

                vertices.push_back(make_float3(std::stof(string_x), std::stof(string_y), std::stof(string_z)));
            }
        }
        obj_file.close();
    }

    return Object3d(triangles, material);
}