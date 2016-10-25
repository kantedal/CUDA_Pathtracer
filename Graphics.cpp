//
// Created by Filip K on 24/10/16.
//

#include <cstdlib>
#include <iostream>
#include "Graphics.h"
#include "kernels/RenderKernel.h"

GLuint vbo;
float3* pixels;

void RenderSample(float3* pixels);

Graphics::Graphics(int *argc, char **argv) {
    if (!InitGL(argc, argv)) return;
    cudaGLSetGLDevice(0);
}

bool Graphics::InitGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Raytracer CUDA");
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, window_width, 0.0, window_height);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glewInit();
    //Timer(0);
    CreateVBO();
    glutMainLoop();

    return true;
}

void Graphics::CreateVBO() {
    //create vertex buffer object
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *&vbo);

    //initialize VBO
    unsigned int size = window_width * window_height * sizeof(float3);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //register VBO with CUDA
    cudaGLRegisterBufferObject(*&vbo);
}

bool Graphics::CheckHW(char *name, const char *gpuType, int dev) {
    return false;
}

int Graphics::FindGraphicsGPU(char *name) {
    return 0;
}

void display() {
    cudaThreadSynchronize();
    cudaGLMapBufferObject((void**)&pixels, vbo);
    glClear(GL_COLOR_BUFFER_BIT);

    RenderSample(pixels);

    cudaThreadSynchronize();
    cudaGLUnmapBufferObject(vbo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, 12, 0);
    glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, window_width * window_height);
    glDisableClientState(GL_VERTEX_ARRAY);
    glutSwapBuffers();

    glutPostRedisplay();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            exit(0);
    }
}
