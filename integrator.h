#include "math_utils.h"

__device__ void integrator(const Scene *s, Ray &ray_, Vector3f &L, int max_depth, curandState *rand_state){
    Vector3f throughput = Vector3f(1, 1, 1);
    Vector3f Le = Vector3f(0, 0, 0);

    Ray r = ray_;

    for(int depth=0; depth < max_depth; depth++){
        SurfaceInteractionRecord si;

        s->intersect(r, 1e-4f, FLT_MAX, si);
        if(!si.hit) break;

        L = L + throughput * si.Le;
        throughput *= si.bsdf;

        Vector3f outward_normal = si.n;
        // The shading normal must always face the incoming ray.
        Vector3f shading_normal = dot(r.direction(), outward_normal) < 0 ? outward_normal : -outward_normal;

        // Calculate the next bounce direction using the corrected shading normal.
        Vector3f target = si.p + shading_normal + random_in_unit_sphere(rand_state);
        Vector3f dir = target - si.p;

        // Create the new ray, offsetting from the surface along the corrected shading normal.
        r = Ray(si.p + shading_normal * 1e-4f, unit_vector(dir));
    }

}


__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(unsigned char* image, Sensor* cam, const Scene *s, curandState *rand_state, int spp) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cam->image_width || y >= cam->image_height) return;

    int pixel_index = y*(cam->image_width) + x;

    float u = float(x + curand_uniform(rand_state)) / float(cam->image_width);
    float v = float(y + curand_uniform(rand_state)) / float(cam->image_height);

    // Get ray for pixel (you need a get_ray method in camera)
    Ray r = cam->get_ray(u,v);


    curandState *local_rand_state = &rand_state[pixel_index];
    Vector3f col = Vector3f(0, 0, 0);
    for (int m = 0; m < spp; m++) {
        float u = float(x + curand_uniform(local_rand_state)) / float(cam->image_width);
        float v = float(y + curand_uniform(local_rand_state)) / float(cam->image_height);

        Ray r = cam->get_ray(u, v);
        
        Vector3f sample_col = Vector3f(0, 0, 0);
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