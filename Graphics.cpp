//
// Created by Filip K on 24/10/16.
//

#include <cstdlib>
#include <iostream>
#include "Graphics.h"
#include "kernels/RenderKernel.h"
#include "kernels/cutil_math.h"

GLuint vbo;
float3* pixels;
float3* clrPixels;
float3* h_clrPixels = new float3[window_height*window_height];
bool cancel = false;

void RenderSample(float3* pixels, float3* clr_pixels);

Graphics::Graphics(int *argc, char **argv) {
    if (!InitGL(argc, argv)) return;
    cudaGLSetGLDevice(0);
}

bool Graphics::InitGL(int *argc, char **argv) {
    cudaMalloc(&clrPixels, window_width * window_height * sizeof(float3));

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

    RenderSample(pixels, clrPixels);

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

    if (!cancel)
        glutPostRedisplay();
}

inline int toInt(float x){ return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27):
            cancel = true;
            cudaMemcpy(h_clrPixels, clrPixels, window_width * window_height * sizeof(float3), cudaMemcpyDeviceToHost);
            cudaFree(pixels);

            FILE *f = fopen("render.ppm", "w");
            fprintf(f, "P3\n%d %d\n%d\n", window_width, window_height, 255);
            for (int i = 0; i < window_width * window_height; i++) {
                fprintf(f, "%d %d %d ", toInt(h_clrPixels[i].x), toInt(h_clrPixels[i].y), toInt(h_clrPixels[i].z));
            }

            printf("Saved image to 'render.ppm'\n");
            exit(0);
    }
}
