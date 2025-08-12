#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "kernels.h"

int main() {
    const int width = 100, height = 100;
    size_t img_size = width * height * 3;

    unsigned char *d_image;
    unsigned char *h_image = new unsigned char[img_size];

    cudaMalloc((void**)&d_image, img_size);

    // Call the CUDA kernel launcher
    launch_write_pixel(d_image, width, height);

    // Copy back to host
    cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost);

    // Save image
    stbi_write_png("red.png", width, height, 3, h_image, width * 3);
    std::cout << "Saved red.png\n";

    cudaFree(d_image);
    delete[] h_image;

    return 0;
}
