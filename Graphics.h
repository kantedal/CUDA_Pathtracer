//
// Created by Filip K on 24/10/16.
//

#ifndef RAYTRACER_CUDA_GRAPHICS_H
#define RAYTRACER_CUDA_GRAPHICS_H

#include <GL/glew.h>
#include <GLUT/glut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

const unsigned int window_width  = 512;
const unsigned int window_height = 512;

void display();
void keyboard(unsigned char key, int /*x*/, int /*y*/);

class Graphics {
public:
    Graphics(int *argc, char **argv);
    Graphics() {};

    bool InitGL(int *argc, char **argv);
    void CreateVBO();
    bool CheckHW(char *name, const char *gpuType, int dev);
    int FindGraphicsGPU(char *name);
private:
};


#endif //RAYTRACER_CUDA_GRAPHICS_H
