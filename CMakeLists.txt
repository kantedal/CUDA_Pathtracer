
project(Raytracer_CUDA)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)


include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})


set(HEADERS
        kernels/Triangle.h
        kernels/Material.h
        kernels/Camera.h
        kernels/Sphere2.h
        kernels/Ray.h
        kernels/RenderKernel.h
        Graphics.h
        Object3d.h
        Scene.h
        RenderDriver.h
        )

set(SOURCES
        Graphics.cpp
        Object3d.cpp
        Scene.cpp
        RenderDriver.cpp
        main.cpp
        kernels/RenderKernel.cu
        kernels/Triangle.h
        )

set(CMAKE_BINARY_DIR   ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH    ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR})

cuda_add_executable(Raytracer_CUDA ${HEADERS} ${SOURCES})

target_link_libraries(Raytracer_CUDA ${CUDA_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})