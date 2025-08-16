#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>
#include <array>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "vec3.h"
#include "ray.h"
#include "sensor.h"
#include "triangle.h"
#include "scene.h"
#include "file_manager.h"
#include "integrator.h"
#include "surface_interaction_record.h"

#ifndef FLT_MAX
#    define FLT_MAX 2e+30
#endif

#include "scene.h" // Your scene management header
#include <iostream>

int main(int argc, char** argv) {
    int image_width = 2000;
    int image_height = 2000;
    int spp = 1000;
    std::string scene_file = "./scenes/cbox.obj";

    if (argc >= 3) {
        image_width = std::stoi(argv[1]);
        image_height = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        spp = std::stoi(argv[3]);
    }

        // --- Extract scene base name ---
    std::string scene_base = scene_file;
    size_t last_slash = scene_base.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        scene_base = scene_base.substr(last_slash + 1);
    }
    size_t last_dot = scene_base.find_last_of(".");
    if (last_dot != std::string::npos) {
        scene_base = scene_base.substr(0, last_dot);
    }

    // --- Build output filename ---
    std::string filename = scene_base + "_" +
                           std::to_string(image_width) + "x" +
                           std::to_string(image_height) + "_" +
                           std::to_string(spp) + ".png";

    std::cout << "Scene file: " << scene_file << "\n";
    std::cout << "Resolution: " << image_width << "x" << image_height << "\n";
    std::cout << "Samples per pixel: " << spp << "\n";
    std::cout << "Output file: " << filename << "\n";

    size_t img_size = image_width * image_height * 3;
    
    unsigned char *d_image;
    unsigned char *h_image = new unsigned char[img_size];

    checkCudaErrors(cudaMalloc((void**)&d_image, img_size * sizeof(unsigned char)));

    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x,
              (image_height + block.y - 1) / block.y);

    Vector3f camera_origin = Vector3f(0.5, 3, 8.5); // Adjusted for a standard cbox view
    Vector3f look_at = Vector3f(0, 2.5, 0);

    Sensor cam(camera_origin, look_at, Vector3f(0,-1,0), 40.0, float(image_width)/float(image_height));
    cam.image_width = image_width;
    cam.image_height = image_height;

    Sensor *d_cam;
    checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(Sensor)));
    checkCudaErrors(cudaMemcpy(d_cam, &cam, sizeof(Sensor), cudaMemcpyHostToDevice));

    int num_pixels = image_width * image_height;
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    Triangle* h_list = nullptr;
    int num_triangles = 0;

    if (!loadOBJ(scene_file, &h_list, num_triangles)) {
        std::cerr << "Failed to load scene. Aborting." << std::endl;
        // Clean up allocated memory before exiting on error
        delete[] h_image;
        cudaFree(d_image);
        cudaFree(d_cam);
        cudaFree(d_rand_state);
        return -1;
    }

    Triangle* d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_triangles * sizeof(Triangle)));
    // Copy directly from the pinned host memory to the device
    checkCudaErrors(cudaMemcpy(d_list, h_list, num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice));

    Scene h_scene(h_list, num_triangles);

    Scene* d_scene;
    checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(Scene)));
    checkCudaErrors(cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(Scene)));
    checkCudaErrors(cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice));

    render_init<<<grid, block>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Rendering with " << spp << " samples per pixel..." << std::endl;
    render<<<grid, block>>>(d_image, d_cam, d_scene, d_rand_state, spp);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost));

    stbi_write_png(filename.c_str(), image_width, image_height, 3, h_image, image_width * 3);
    std::cout << "Image saved as " << filename << "\n";

    // --- Cleanup ---
    cudaFree(d_image);
    cudaFree(d_cam);
    cudaFree(d_list);
    cudaFree(d_scene);
    cudaFree(d_rand_state);
    // Use cudaFreeHost for memory allocated with cudaMallocHost
    checkCudaErrors(cudaFreeHost(h_list));

    delete[] h_image;
    cudaDeviceReset();

    return 0;
}
