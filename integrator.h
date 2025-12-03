#include "math_utils.h"
#include "sampling_utils.h"
#include "render_config.h"

// Use grid size from render_config.h (DEFAULT_GRID_RES = 50)
#ifndef GRID_RES
#define GRID_RES DEFAULT_GRID_RES
#endif
#define RES GRID_RES
#ifndef GRID_SIZE
#define GRID_SIZE (GRID_RES * GRID_RES)
#endif

// ==================== OPTIMIZED GRID SAMPLING WITH CACHING ====================

struct SurfaceSamplingData {
    float cdf_x[GRID_RES];
    float cdf_y[GRID_RES];
    float total_weight;
    bool is_valid;
};

// Keep old interface for compatibility, use fast utilities
__device__ void buildGridPDF(const float* grid, SurfaceSamplingData &sampling_data) {
    float total;
    buildGridPDFFast(grid, GRID_RES, sampling_data.cdf_y, total);
    sampling_data.total_weight = total;
    sampling_data.is_valid = (total > 1e-6f);
}

__device__ void buildConditionalCDF(const float* grid, int v_idx, float* cdf_x) {
    buildConditionalCDFFast(grid, v_idx, GRID_RES, cdf_x);
}

__device__ inline int binarySearchCDF(const float* cdf, int size, float xi) {
    return binarySearchCDFFast(cdf, size, xi);
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


// ==================== GRID-BASED DIRECTION SAMPLING ====================

// ==================== OPTIMIZED GRID SAMPLING ====================

// Sample from form factor grid using SPHERICAL coordinates with PDF output
// Only samples from upper hemisphere (theta in [0, π/2])
__device__ inline Vector3f sampleGridDirectionWithPDF(const float* grid, const Vector3f &normal, 
                                        curandState *rand_state, bool &success, float &pdf_out) {
    // Build CDF only for upper hemisphere
    int half_res = GRID_RES / 2;
    float row_sums[GRID_RES];
    float cdf_y[GRID_RES];
    float hemisphere_weight = 0.0f;
    
    for (int v = 0; v < half_res; v++) {
        float row_sum = 0.0f;
        for (int u = 0; u < GRID_RES; u++) {
            row_sum += grid[v * GRID_RES + u];
        }
        row_sums[v] = row_sum;
        hemisphere_weight += row_sum;
        cdf_y[v] = hemisphere_weight;
    }
    
    if (hemisphere_weight < 1e-6f) {
        success = false;
        Vector3f dir = sampleCosineWeightedHemisphereFast(normal, rand_state);
        pdf_out = cosinePDF(dir, normal);
        return dir;
    }
    
    // Normalize CDF
    for (int v = 0; v < half_res; v++) {
        cdf_y[v] /= hemisphere_weight;
    }
    
    // Sample row (theta index) from upper hemisphere only
    float xi1 = curand_uniform(rand_state);
    int theta_idx = binarySearchCDF(cdf_y, half_res, xi1);
    
    // Sample column (phi index)
    float cdf_x[GRID_RES];
    float row_sum = row_sums[theta_idx];
    if (row_sum < 1e-6f) {
        for (int u = 0; u < GRID_RES; u++) cdf_x[u] = (u + 1.0f) / GRID_RES;
    } else {
        float rs = 0.0f;
        for (int u = 0; u < GRID_RES; u++) {
            rs += grid[theta_idx * GRID_RES + u];
            cdf_x[u] = rs / row_sum;
        }
    }
    
    float xi2 = curand_uniform(rand_state);
    int phi_idx = binarySearchCDF(cdf_x, GRID_RES, xi2);
    
    // Convert grid indices to SPHERICAL coordinates (hemisphere only)
    float theta = ((theta_idx + curand_uniform(rand_state)) / half_res) * (M_PI * 0.5f);
    theta = fminf(theta, M_PI * 0.5f - 0.01f);  // Stay away from horizon
    float phi = ((phi_idx + curand_uniform(rand_state)) / GRID_RES) * 2.0f * M_PI;
    
    // Convert spherical to Cartesian in local space
    float sin_theta = sinf(theta);
    float cos_theta = cosf(theta);
    Vector3f local_dir(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
    
    // Transform to world space
    Vector3f tangent, bitangent;
    buildCoordinateSystemFast(normal, tangent, bitangent);
    Vector3f world_dir = unit_vector(localToWorldFast(local_dir, normal, tangent, bitangent));
    
    // Compute the grid sampling PDF (hemisphere-adjusted)
    float cell_value = grid[theta_idx * GRID_RES + phi_idx];
    float cell_prob = cell_value / fmaxf(hemisphere_weight, 1e-6f);
    float d_theta = (M_PI * 0.5f) / half_res;
    float d_phi = 2.0f * M_PI / GRID_RES;
    float cell_solid_angle = fmaxf(sin_theta, 0.01f) * d_theta * d_phi;
    pdf_out = cell_prob / fmaxf(cell_solid_angle, 1e-6f);
    
    success = true;
    return world_dir;
}

// Legacy version without PDF (for compatibility)
__device__ inline Vector3f sampleGridDirection(const float* grid, const Vector3f &normal, 
                                        curandState *rand_state, bool &success) {
    float pdf_unused;
    return sampleGridDirectionWithPDF(grid, normal, rand_state, success, pdf_unused);
}

// ==================== RADIOSITY DIRECTION SAMPLING ====================

// Sample from radiosity grid with PDF output
__device__ inline Vector3f sampleRadiosityDirectionWithPDF(const Vector3f* radiosity_grid, const Vector3f &normal,
                                            curandState *rand_state, bool &success, float &pdf_out) {
    float total_weight = computeRadiosityTotalWeight(radiosity_grid, GRID_RES);
    
    if (total_weight < 1e-6f) {
        success = false;
        Vector3f dir = sampleCosineWeightedHemisphereFast(normal, rand_state);
        pdf_out = cosinePDF(dir, normal);
        return dir;
    }
    
    return sampleRadiosityGridSpherical(radiosity_grid, GRID_RES, normal, rand_state, pdf_out, total_weight);
}

__device__ inline Vector3f sampleRadiosityDirection(const Vector3f* radiosity_grid, const Vector3f &normal,
                                            curandState *rand_state, bool &success) {
    float pdf_unused;
    Vector3f dir = sampleRadiosityDirectionWithPDF(radiosity_grid, normal, rand_state, success, pdf_unused);
    return dir;
}

// ==================== PATH TRACER ====================

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
        bool use_grid = (sampling_mode == SamplingMode::SAMPLING_FORMFACTOR || 
                         sampling_mode == SamplingMode::SAMPLING_RADIOSITY ||
                         sampling_mode == SamplingMode::SAMPLING_TOPK ||
                         sampling_mode == SamplingMode::SAMPLING_MIS);
        bool has_grid = false;
        const float* grid = nullptr;
        const Vector3f* radiosity_grid = nullptr;
        
        if (use_grid && si.prim_ptr != nullptr) {
            // Always get radiosity grid for MIS and RADIOSITY modes
            if (sampling_mode == SamplingMode::SAMPLING_RADIOSITY || 
                sampling_mode == SamplingMode::SAMPLING_MIS) {
                if (si.hit_type == HIT_TRIANGLE) {
                    radiosity_grid = si.prim_ptr->tri.radiosity_grid;
                } else if (si.hit_type == HIT_QUAD) {
                    radiosity_grid = si.prim_ptr->quad.radiosity_grid;
                }
                has_grid = (radiosity_grid != nullptr);
            } else {
                // Use form factor grid for FORMFACTOR and TOPK modes
                if (si.hit_type == HIT_TRIANGLE) {
                    grid = si.prim_ptr->tri.grid;
                } else if (si.hit_type == HIT_QUAD) {
                    grid = si.prim_ptr->quad.grid;
                }
                has_grid = (grid != nullptr);
            }
        }
        
        // Sample direction based on mode
        if (sampling_mode == SamplingMode::SAMPLING_BSDF || !has_grid) {
            next_dir = sampleCosineWeightedHemisphereFast(shading_normal, rand_state);
        }
        else if (sampling_mode == SamplingMode::SAMPLING_FORMFACTOR) {
            bool success;
            float grid_pdf;
            next_dir = sampleGridDirectionWithPDF(grid, shading_normal, rand_state, success, grid_pdf);
            if (!success) {
                next_dir = sampleCosineWeightedHemisphereFast(shading_normal, rand_state);
            } else {
                // PDF correction: weight = (cosine_pdf) / (grid_pdf)
                // cosine_pdf = cos(theta) / pi
                float cos_theta = fmaxf(dot(next_dir, shading_normal), 0.0f);
                float cosine_pdf = cos_theta / M_PI;
                float weight = cosine_pdf / fmaxf(grid_pdf, 1e-6f);
                weight = fminf(weight, 10.0f);  // Clamp to prevent fireflies
                throughput *= weight;
            }
        }
        else if (sampling_mode == SamplingMode::SAMPLING_RADIOSITY) {
            bool success;
            float grid_pdf;
            next_dir = sampleRadiosityDirectionWithPDF(radiosity_grid, shading_normal, rand_state, success, grid_pdf);
            if (!success) {
                next_dir = sampleCosineWeightedHemisphereFast(shading_normal, rand_state);
            } else {
                // PDF correction: weight = (cosine_pdf) / (grid_pdf)
                float cos_theta = fmaxf(dot(next_dir, shading_normal), 0.0f);
                float cosine_pdf = cos_theta / M_PI;
                float weight = cosine_pdf / fmaxf(grid_pdf, 1e-6f);
                weight = fminf(weight, 10.0f);  // Clamp to prevent fireflies
                throughput *= weight;
            }
        }
        else if (sampling_mode == SamplingMode::SAMPLING_MIS) {
            // Use radiosity-based MIS (correct version)
            next_dir = sampleMISFastRadiosity(radiosity_grid, GRID_RES, shading_normal, rand_state, mis_weight);
            throughput *= mis_weight;
        }
        else { // SAMPLING_TOPK - use form factor grid
            next_dir = sampleMISFast(grid, GRID_RES, shading_normal, rand_state, mis_weight);
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
        Vector3f next_dir = sampleCosineWeightedHemisphereFast(shading_normal, rand_state);

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
        integrator(s, r, sample_col, 5, local_rand_state, sampling_mode);
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
