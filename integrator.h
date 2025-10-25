#include "math_utils.h"

// Match the grid size from triangle.h and quad.h
#define GRID_RES 20  // Changed from 20 to 100
#define RES GRID_RES
#define GRID_SIZE (GRID_RES * GRID_RES)  // 10000

// ==================== SAMPLING MODE ENUM ====================

enum SamplingMode {
    SAMPLING_BSDF = 0,
    SAMPLING_GRID = 1,
    SAMPLING_MIS = 2
};

// ==================== OPTIMIZED GRID SAMPLING WITH CACHING ====================

struct SurfaceSamplingData {
    float cdf_x[GRID_RES];      // Single row CDF for faster lookup
    float cdf_y[GRID_RES];      // Marginal CDF
    float total_weight;
    bool is_valid;
};

// Helper to access 1D grid as 2D
__device__ inline float getGridValue(const float* grid, int i, int j) {
    return grid[i * GRID_RES + j];
}

// OPTIMIZED: Faster PDF building with early exit and reduced memory
__device__ void buildGridPDF(const float* grid, SurfaceSamplingData &sampling_data) {
    float row_sums[GRID_RES];
    float total = 0.0f;
    
    // Single pass to compute row sums and total
    #pragma unroll 4
    for (int v = 0; v < GRID_RES; v++) {
        float row_sum = 0.0f;
        #pragma unroll 4
        for (int u = 0; u < GRID_RES; u++) {
            row_sum += getGridValue(grid, v, u);
        }
        row_sums[v] = row_sum;
        total += row_sum;
    }
    
    // Early exit for empty grids
    if (total < 1e-6f) {
        sampling_data.is_valid = false;
        return;
    }
    
    sampling_data.total_weight = total;
    sampling_data.is_valid = true;
    
    // Build marginal CDF
    float running_sum = 0.0f;
    for (int v = 0; v < GRID_RES; v++) {
        sampling_data.cdf_y[v] = running_sum / total;
        running_sum += row_sums[v];
    }
}

// OPTIMIZED: Build conditional CDF on-demand for selected row only
__device__ void buildConditionalCDF(const float* grid, int v_idx, float* cdf_x) {
    float row_sum = 0.0f;
    
    // Compute row sum
    for (int u = 0; u < GRID_RES; u++) {
        row_sum += getGridValue(grid, v_idx, u);
    }
    
    if (row_sum < 1e-6f) {
        // Uniform distribution for empty row
        for (int u = 0; u < GRID_RES; u++) {
            cdf_x[u] = float(u) / float(GRID_RES);
        }
        return;
    }
    
    // Build CDF
    float running_sum = 0.0f;
    for (int u = 0; u < GRID_RES; u++) {
        cdf_x[u] = running_sum / row_sum;
        running_sum += getGridValue(grid, v_idx, u);
    }
}

// OPTIMIZED: Binary search with bounds checking
__device__ inline int binarySearchCDF(const float* cdf, int size, float xi) {
    // Clamp xi to valid range
    xi = fminf(fmaxf(xi, 0.0f), 0.999999f);
    
    int left = 0;
    int right = size - 1;
    
    while (left < right) {
        int mid = (left + right) >> 1;  // Faster than division
        if (xi < cdf[mid + 1]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

// OPTIMIZED: Grid sampling with on-demand conditional CDF
__device__ Vector2f sampleGridPDF(const float* grid, const SurfaceSamplingData &sampling_data, 
                                  curandState *rand_state) {
    float xi1 = curand_uniform(rand_state);
    float xi2 = curand_uniform(rand_state);
    
    // Sample row using marginal CDF
    int v_idx = binarySearchCDF(sampling_data.cdf_y, GRID_RES, xi1);
    
    // Build conditional CDF only for this row
    float cdf_x[GRID_RES];
    buildConditionalCDF(grid, v_idx, cdf_x);
    
    // Sample column
    int u_idx = binarySearchCDF(cdf_x, GRID_RES, xi2);
    
    // Stratified jitter for smoother sampling
    float u = (u_idx + curand_uniform(rand_state)) / float(GRID_RES);
    float v = (v_idx + curand_uniform(rand_state)) / float(GRID_RES);
    
    return Vector2f(u, v);
}

// IMPROVED: Better UV to direction with proper hemisphere mapping
__device__ Vector3f uvToDirection(float u, float v) {
    // Cosine-weighted hemisphere sampling using Malley's method
    float r = sqrtf(u);
    float theta = 2.0f * M_PI * v;
    
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u));  // Ensure non-negative
    
    return Vector3f(x, y, z);
}

// OPTIMIZED: Robust coordinate system building
__device__ void buildCoordinateSystem(const Vector3f &n, Vector3f &tangent, Vector3f &bitangent) {
    // Frisvad's method - branchless and numerically stable
    if (n.z() < -0.9999999f) {
        tangent = Vector3f(0.0f, -1.0f, 0.0f);
        bitangent = Vector3f(-1.0f, 0.0f, 0.0f);
        return;
    }
    
    float a = 1.0f / (1.0f + n.z());
    float b = -n.x() * n.y() * a;
    
    tangent = Vector3f(1.0f - n.x() * n.x() * a, b, -n.x());
    bitangent = Vector3f(b, 1.0f - n.y() * n.y() * a, -n.y());
}

__device__ Vector3f localToWorld(const Vector3f &local, const Vector3f &normal, 
                                  const Vector3f &tangent, const Vector3f &bitangent) {
    return tangent * local.x() + bitangent * local.y() + normal * local.z();
}

// ==================== IMPROVED BSDF SAMPLING ====================

__device__ Vector3f sampleCosineWeightedHemisphere(const Vector3f &normal, curandState *rand_state) {
    // Malley's method for cosine-weighted hemisphere sampling
    float u1 = curand_uniform(rand_state);
    float u2 = curand_uniform(rand_state);
    
    float r = sqrtf(u1);
    float theta = 2.0f * M_PI * u2;
    
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    
    Vector3f tangent, bitangent;
    buildCoordinateSystem(normal, tangent, bitangent);
    
    return unit_vector(tangent * x + bitangent * y + normal * z);
}

// ==================== GRID-BASED DIRECTION SAMPLING ====================

__device__ Vector3f sampleGridDirection(const float* grid, const Vector3f &normal, 
                                        curandState *rand_state, bool &success) {
    SurfaceSamplingData sampling_data;
    buildGridPDF(grid, sampling_data);
    
    if (!sampling_data.is_valid) {
        success = false;
        return Vector3f(0, 0, 0);
    }
    
    Vector2f uv = sampleGridPDF(grid, sampling_data, rand_state);
    Vector3f local_dir = uvToDirection(uv.x(), uv.y());
    
    Vector3f tangent, bitangent;
    buildCoordinateSystem(normal, tangent, bitangent);
    
    Vector3f world_dir = localToWorld(local_dir, normal, tangent, bitangent);
    success = true;
    return unit_vector(world_dir);
}

// ==================== IMPROVED MIS ====================

__device__ Vector3f sampleMIS(const float* grid, const Vector3f &normal, 
                              curandState *rand_state, float &weight) {
    float sample_choice = curand_uniform(rand_state);
    
    // Try grid sampling first if available
    if (sample_choice < 0.5f) {
        bool success;
        Vector3f dir = sampleGridDirection(grid, normal, rand_state, success);
        if (success) {
            weight = 2.0f;
            return dir;
        }
    }
    
    // BSDF sampling
    weight = 2.0f;
    return sampleCosineWeightedHemisphere(normal, rand_state);
}

// ==================== OPTIMIZED PATH TRACER ====================

__device__ void integrator(const Scene *s, Ray &ray_, Vector3f &L, int max_depth, 
                          curandState *rand_state, SamplingMode sampling_mode) {
    Vector3f throughput = Vector3f(1, 1, 1);
    Ray r = ray_;
    
    for(int depth = 0; depth < max_depth; depth++) {
        SurfaceInteractionRecord si;
        if(!s->intersect(r, 1e-4f, FLT_MAX, si)) {
            break;
        }
        
        L += throughput * si.Le;
        
        // Aggressive Russian roulette for faster termination
        if(depth > 2) {
            float q = fmaxf(0.05f, 1.0f - fmaxf(throughput.x(), fmaxf(throughput.y(), throughput.z())));
            if(curand_uniform(rand_state) < q) {
                break;
            }
            throughput /= (1.0f - q);
        }
        
        throughput *= si.bsdf;
        
        // Early termination for negligible contribution
        if(throughput.length() < 1e-4f) {
            break;
        }
        
        Vector3f outward_normal = si.n;
        Vector3f shading_normal = dot(r.direction(), outward_normal) < 0 ? outward_normal : -outward_normal;
        
        Vector3f next_dir;
        float mis_weight = 1.0f;
        
        // Branch-reduced sampling selection
        bool use_grid = (sampling_mode == SAMPLING_GRID || sampling_mode == SAMPLING_MIS);
        bool has_grid = false;
        const float* grid = nullptr;
        
        if (use_grid && si.prim_ptr != nullptr) {
            if (si.hit_type == HIT_TRIANGLE) {
                grid = si.prim_ptr->tri.grid;
            } else if (si.hit_type == HIT_QUAD) {
                grid = si.prim_ptr->quad.grid;
            }
            has_grid = (grid != nullptr);
        }
        
        // Sample direction based on mode
        if (sampling_mode == SAMPLING_BSDF || !has_grid) {
            next_dir = sampleCosineWeightedHemisphere(shading_normal, rand_state);
        }
        else if (sampling_mode == SAMPLING_GRID) {
            bool success;
            next_dir = sampleGridDirection(grid, shading_normal, rand_state, success);
            if (!success) {
                next_dir = sampleCosineWeightedHemisphere(shading_normal, rand_state);
            }
        }
        else { // SAMPLING_MIS
            next_dir = sampleMIS(grid, shading_normal, rand_state, mis_weight);
            throughput *= mis_weight;
        }
        
        r = Ray(si.p + shading_normal * 1e-4f, next_dir);
    }
}

// ==================== RADIOSITY INTEGRATOR ====================

__device__ void radiosity_integrator(const Scene *s, Ray &ray_, Vector3f &L, int max_depth, curandState *rand_state){
    Vector3f throughput = Vector3f(1, 1, 1);
    Ray r = ray_;

    for(int depth = 0; depth < max_depth; depth++){
        SurfaceInteractionRecord si;

        if(!s->intersect(r, 1e-4f, FLT_MAX, si)) {
            break;
        }

        L += throughput * (si.Le + si.radiosity);

        if(depth == 0) {
            break;
        }

        throughput *= si.bsdf;

        Vector3f outward_normal = si.n;
        Vector3f shading_normal = dot(r.direction(), outward_normal) < 0 ? outward_normal : -outward_normal;

        // Use improved cosine-weighted sampling
        Vector3f next_dir = sampleCosineWeightedHemisphere(shading_normal, rand_state);

        r = Ray(si.p + shading_normal * 1e-4f, next_dir);
    }
}

// ==================== RENDER KERNELS ====================

__global__ void render_init(int width, int height, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    curand_init(2023 + pixel_index, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(unsigned char* image, Sensor* cam, Scene *s, curandState *rand_state, 
                      int spp, SamplingMode sampling_mode) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cam->image_width || y >= cam->image_height) return;

    int pixel_index = y * cam->image_width + x;
    curandState *local_rand_state = &rand_state[pixel_index];

    Vector3f col = Vector3f(0, 0, 0);
    
    for (int m = 0; m < spp; m++) {
        float u = (x + curand_uniform(local_rand_state)) / float(cam->image_width);
        float v = (y + curand_uniform(local_rand_state)) / float(cam->image_height);

        Ray r = cam->get_ray(u, v);
        
        Vector3f sample_col = Vector3f(0, 0, 0);
        integrator(s, r, sample_col, 10, local_rand_state, sampling_mode);
        col += sample_col;
    }

    col /= float(spp);

    // Improved tone mapping (ACES approximation)
    col = col / (col + Vector3f(1, 1, 1));

    // Gamma correction
    const float gamma = 1.0f / 2.2f;
    col.e[0] = powf(col.e[0], gamma);
    col.e[1] = powf(col.e[1], gamma);
    col.e[2] = powf(col.e[2], gamma);

    int idx = (y * cam->image_width + x) * 3;
    image[idx + 0] = static_cast<unsigned char>(255.99f * fminf(col.r(), 1.0f));
    image[idx + 1] = static_cast<unsigned char>(255.99f * fminf(col.g(), 1.0f));
    image[idx + 2] = static_cast<unsigned char>(255.99f * fminf(col.b(), 1.0f));
}

__global__ void render_radiosity(unsigned char* image, Sensor* d_cam, Scene* d_world,
                                 curandState* rand_state, int spp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= d_cam->image_width || j >= d_cam->image_height) return;
    
    int pixel_index = j * d_cam->image_width + i;
    curandState local_rand_state = rand_state[pixel_index];
    
    Vector3f pixel_color(0.0f, 0.0f, 0.0f);
    
    for (int s = 0; s < spp; s++) {
        float u = (float(i) + curand_uniform(&local_rand_state)) / float(d_cam->image_width);
        float v = (float(j) + curand_uniform(&local_rand_state)) / float(d_cam->image_height);
        
        Ray r = d_cam->get_ray(u, v);
        SurfaceInteractionRecord si;
        
        if (d_world->intersect(r, 1e-4f, FLT_MAX, si)) {
            pixel_color += si.Le;
            
            if (si.prim_ptr != nullptr) {
                Vector3f rad = si.prim_ptr->getRadiosity();
                pixel_color += rad;
            } else {
                pixel_color += si.radiosity;
            }
        }
    }
    
    pixel_color = pixel_color / float(spp);
    
    pixel_color = Vector3f(
        sqrtf(fminf(pixel_color.x(), 1.0f)),
        sqrtf(fminf(pixel_color.y(), 1.0f)),
        sqrtf(fminf(pixel_color.z(), 1.0f))
    );
    
    int index = 3 * pixel_index;
    image[index + 0] = (unsigned char)(255.99f * pixel_color.x());
    image[index + 1] = (unsigned char)(255.99f * pixel_color.y());
    image[index + 2] = (unsigned char)(255.99f * pixel_color.z());
    
    rand_state[pixel_index] = local_rand_state;
}

__global__ void radiosity_delta_integrator(
    unsigned char* image, int width, int height,
    Sensor* camera, Scene* scene, int step1, int step2) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;
    
    Vector3f direction = unit_vector(
        camera->lower_left_corner + u * camera->horizontal +
        v * camera->vertical - camera->origin);
    Ray r(camera->origin, direction);
    
    SurfaceInteractionRecord si;
    
    if (scene->intersect(r, 1e-4f, FLT_MAX, si)) {
        if (si.prim_ptr != nullptr) {
            Vector3f rad_step1 = si.prim_ptr->getRadiosityHistory(step1);
            Vector3f rad_step2 = si.prim_ptr->getRadiosityHistory(step2);
            
            Vector3f delta = rad_step1 - rad_step2;
            Vector3f color = delta * 5.0f;
            
            int idx = (y * width + x) * 3;
            image[idx + 0] = (unsigned char)fminf(fmaxf(color.x() * 255.0f, 0.0f), 255.0f);
            image[idx + 1] = (unsigned char)fminf(fmaxf(color.y() * 255.0f, 0.0f), 255.0f);
            image[idx + 2] = (unsigned char)fminf(fmaxf(color.z() * 255.0f, 0.0f), 255.0f);
        } else {
            int idx = (y * width + x) * 3;
            image[idx + 0] = 255;
            image[idx + 1] = 0;
            image[idx + 2] = 0;
        }
    } else {
        int idx = (y * width + x) * 3;
        image[idx + 0] = image[idx + 1] = image[idx + 2] = 0;
    }
}
