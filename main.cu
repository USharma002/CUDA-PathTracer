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
#include "camera.h"
#include "triangle.h"
#include "scene.h"
#include "file_manager.h"
#include "surface_interaction_record.h"


#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ void integrator(const scene *s, ray &ray_, vec3 &L, int max_depth, curandState *rand_state){
    vec3 throughput = vec3(1, 1, 1);
    vec3 Le = vec3(0, 0, 0);

    ray r = ray_;

    for(int i=0; i < max_depth; i++){
        surface_interaction_record si;

        s->intersect(r, 1e-4f, FLT_MAX, si);
        if(!si.hit) break;

        L = L + throughput * si.Le;
        throughput *= si.bsdf;

        vec3 outward_normal = si.n;
        // The shading normal must always face the incoming ray.
        vec3 shading_normal = dot(r.direction(), outward_normal) < 0 ? outward_normal : -outward_normal;

        // Calculate the next bounce direction using the corrected shading normal.
        vec3 target = si.p + shading_normal + random_in_unit_sphere(rand_state);
        vec3 dir = target - si.p;

        // Create the new ray, offsetting from the surface along the corrected shading normal.
        r = ray(si.p + shading_normal * 1e-4f, unit_vector(dir));
    }

    // s->intersect(r, 1e-6, FLT_MAX, si); 
    // L = L + (si.n + vec3(1.0, 1, 1))/vec3(2.0, 2, 2);

}

__global__ void render(unsigned char* image, camera* cam, const scene *s, curandState *rand_state, int spp) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cam->image_width || y >= cam->image_height) return;

    int pixel_index = y*(cam->image_width) + x;

    float u = float(x + curand_uniform(rand_state)) / float(cam->image_width);
    float v = float(y + curand_uniform(rand_state)) / float(cam->image_height);

    // Get ray for pixel (you need a get_ray method in camera)
    ray r = cam->get_ray(u,v);


    curandState *local_rand_state = &rand_state[pixel_index];
    vec3 col = vec3(0, 0, 0);
    for (int m = 0; m < spp; m++) {
        float u = float(x + curand_uniform(local_rand_state)) / float(cam->image_width);
        float v = float(y + curand_uniform(local_rand_state)) / float(cam->image_height);

        ray r = cam->get_ray(u, v);
        
        vec3 sample_col = vec3(0, 0, 0);
        integrator(s, r, sample_col, 10, local_rand_state);
        col += sample_col;
    }

    col /= float(spp);

    col.e[0] = col.e[0] / (col.e[0] + 1.0f);
    col.e[1] = col.e[1] / (col.e[1] + 1.0f);
    col.e[2] = col.e[2] / (col.e[2] + 1.0f);

    // Use powf() for x^(1/2.2)
    const float gamma = 1.0f / 2.2f;
    col.e[0] = powf(col.e[0], gamma);
    col.e[1] = powf(col.e[1], gamma);
    col.e[2] = powf(col.e[2], gamma);

    int idx = (y * cam->image_width + x) * 3;
    image[idx + 0] = static_cast<unsigned char>(255.99f * col.r());
    image[idx + 1] = static_cast<unsigned char>(255.99f * col.g());
    image[idx + 2] = static_cast<unsigned char>(255.99f * col.b());
}


__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// Helper function to set triangle properties and calculate normal
void set_triangle(triangle& tri, const vec3& Le, const vec3& bsdf) {
    tri.Le = Le;
    tri.bsdf = bsdf;
    vec3 edge1 = tri.v1 - tri.v0;
    vec3 edge2 = tri.v2 - tri.v0;
    tri.normal = unit_vector(cross(edge1, edge2));
}


int main() {
    const int image_width = 2000, image_height = 2000;
    size_t img_size = image_width * image_height * 3;
    int spp = 1024;

    unsigned char *d_image;
    unsigned char *h_image = new unsigned char[img_size];

    checkCudaErrors(cudaMalloc((void**)&d_image, img_size * sizeof(unsigned char)));

    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x,
              (image_height + block.y - 1) / block.y);

    // Camera setup remains the same
    vec3 camera_origin = vec3(0.5, 3, 8.5); // Adjusted for a standard cbox view
    vec3 look_at = vec3(0, 2.5, 0);

    camera cam(camera_origin, look_at, vec3(0,-1,0), 40.0, float(image_width)/float(image_height));
    cam.image_width = image_width;
    cam.image_height = image_height;

    camera *d_cam;
    checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera)));
    checkCudaErrors(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice));

    int num_pixels = image_width * image_height;
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    triangle* h_list = nullptr;
    int num_triangles = 0;

    if (!loadOBJ("cbox.obj", &h_list, num_triangles)) {
        std::cerr << "Failed to load scene. Aborting." << std::endl;
        // Clean up allocated memory before exiting on error
        delete[] h_image;
        cudaFree(d_image);
        cudaFree(d_cam);
        cudaFree(d_rand_state);
        return -1;
    }

    triangle* d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_triangles * sizeof(triangle)));
    // Copy directly from the pinned host memory to the device
    checkCudaErrors(cudaMemcpy(d_list, h_list, num_triangles * sizeof(triangle), cudaMemcpyHostToDevice));

    scene h_scene(d_list, num_triangles);
    scene* d_scene;
    checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(scene)));
    checkCudaErrors(cudaMemcpy(d_scene, &h_scene, sizeof(scene), cudaMemcpyHostToDevice));

    render_init<<<grid, block>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Rendering with " << spp << " samples per pixel..." << std::endl;
    render<<<grid, block>>>(d_image, d_cam, d_scene, d_rand_state, spp);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost));

    stbi_write_png("image.png", image_width, image_height, 3, h_image, image_width * 3);
    std::cout << "Image saved as image.png\n";

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