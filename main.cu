#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include "vec3.h"
#include "ray.h"
#include "camera.h"
#include "triangle.h"
#include "surface_interaction_record.h"


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ void integrator(ray* ray_, surface_interaction_record *si, vec3* L, int max_depth){

    for(int i=0; i < max_depth; i++)
    *L = vec3(0, 0, 1);
}

__global__ void render(unsigned char* image, const camera* cam, const triangle* tri) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cam->image_width || y >= cam->image_height) return;

    // Get ray for pixel (you need a get_ray method in camera)
    ray r = cam->get_ray(x, y);

    surface_interaction_record si;
    vec3 col;
    integrator(&r, &si, &col, 5);

    int idx = (y * cam->image_width + x) * 3;
    image[idx + 0] = static_cast<unsigned char>(255.99f * col.r());
    image[idx + 1] = static_cast<unsigned char>(255.99f * col.g());
    image[idx + 2] = static_cast<unsigned char>(255.99f * col.b());
}

int main() {
    const int image_width = 2000, image_height = 2000;
    size_t img_size = image_width * image_height * 3;

    unsigned char *d_image;
    unsigned char *h_image = new unsigned char[img_size];

    // Allocate memory on GPU
    checkCudaErrors(cudaMalloc((void**)&d_image, img_size * sizeof(unsigned char)));

    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x,
              (image_height + block.y - 1) / block.y);

    vec3 camera_origin = {0, 0, 0};

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width)/image_height);

    camera cam(
        camera_origin,
        image_width,
        image_height,
        focal_length,
        viewport_height,
        viewport_width
    );

    camera *d_cam;
    cudaMalloc((void**)&d_cam, sizeof(camera));
    checkCudaErrors(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice));

    triangle tri(
        vec3(-0.5f, -0.5f, -1.0f),
        vec3(0.5f, -0.5f, -1.0f),
        vec3(0.0f, 0.5f, -1.0f)
    );

    // Allocate device memory for triangle
    triangle* d_tri;
    cudaMalloc((void**)&d_tri, sizeof(triangle));
    checkCudaErrors(cudaMemcpy(d_tri, &tri, sizeof(triangle), cudaMemcpyHostToDevice));

    // Launch kernel passing triangle pointer
    render<<<grid, block>>>(d_image, d_cam, d_tri);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy data back to host
    checkCudaErrors(cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost));

    // Save as PNG
    stbi_write_png("image.png", image_width, image_height, 3, h_image, image_width * 3);
    std::cout << "Image saved as image.png\n";

    // Free memory
    cudaFree(d_image);
    cudaFree(d_cam);
    cudaFree(d_tri);

    delete[] h_image;

    cudaDeviceReset();

    return 0;
}
