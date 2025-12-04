/**
 * @file integrator.h
 * @brief Clean path tracer with optional grid-guided sampling
 * 
 * Design modeled after Mitsuba's path integrator:
 * - Simple loop structure  
 * - MIS with power heuristic
 * - Russian roulette for termination
 * 
 * The Grid class handles all sampling details - integrator just calls grid.sample()
 */

#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "core/math_utils.h"
#include "rendering/grid.h"
#include "rendering/render_config.h"

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Initialize Grid from the hit primitive's data
 * Returns true if grid was successfully initialized
 * 
 * OPTIMIZATION: Uses precomputed CDFs when available (no per-sample CDF building!)
 * Precomputed CDFs work for both raw AND filtered radiosity data.
 */
__device__ bool initGridFromPrimitive(const SurfaceInteractionRecord& si, 
                                       const Scene* scene, Grid& grid) {
    if (si.prim_ptr == nullptr) return false;
    
    int prim_idx = (int)(si.prim_ptr - scene->primitives);
    
    // FAST PATH: Use precomputed CDFs if available (works for both raw and filtered!)
    if (scene->precomputed_cdfs != nullptr) {
        grid.loadPrecomputed(&scene->precomputed_cdfs[prim_idx]);
        return grid.isValid();
    }
    
    // SLOW FALLBACK: Use raw radiosity grid (builds CDF every sample)
    const Vector3f* rad_grid = nullptr;
    if (si.hit_type == HIT_TRIANGLE) {
        rad_grid = si.prim_ptr->tri.radiosity_grid;
    } else if (si.hit_type == HIT_QUAD) {
        rad_grid = si.prim_ptr->quad.radiosity_grid;
    }
    
    if (rad_grid != nullptr) {
        grid.initFromRadiosity(rad_grid);
        return true;
    }
    
    return false;
}

/**
 * Cosine-weighted hemisphere sampling (Malley's method)
 */
__device__ Vector3f sampleCosineHemisphere(const Vector3f& normal, curandState* rng) {
    float u = curand_uniform(rng);
    float v = curand_uniform(rng);
    
    float r = sqrtf(u);
    float phi = 2.0f * M_PI * v;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u));
    
    // Build frame (Frisvad's method)
    Vector3f tangent, bitangent;
    if (normal.z() < -0.9999999f) {
        tangent = Vector3f(0.0f, -1.0f, 0.0f);
        bitangent = Vector3f(-1.0f, 0.0f, 0.0f);
    } else {
        float a = 1.0f / (1.0f + normal.z());
        float b = -normal.x() * normal.y() * a;
        tangent = Vector3f(1.0f - normal.x() * normal.x() * a, b, -normal.x());
        bitangent = Vector3f(b, 1.0f - normal.y() * normal.y() * a, -normal.y());
    }
    
    return unit_vector(tangent * x + bitangent * y + normal * z);
}

/**
 * MIS power heuristic weight (following Mitsuba's implementation)
 * Returns: pdf_a² / (pdf_a² + pdf_b²)
 */
__device__ float misPowerHeuristic(float pdf_a, float pdf_b) {
    if (pdf_a <= 0.0f) return 0.0f;
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2);
}

/**
 * MIS sampling: randomly choose grid or BSDF based on selection probability
 * 
 * Uses one-sample MIS with power heuristic (Veach's thesis).
 * Selection probability determines how often we use each technique.
 * MIS weight ensures unbiased combination regardless of selection prob.
 * 
 * @param grid       The sampling grid (importance distribution)
 * @param normal     Surface normal
 * @param rng        Random state
 * @param weight     Output: importance sampling weight for rendering equation
 * @param bsdf_prob  Probability of selecting BSDF sampling (0 to 1)
 * @param used_bsdf  Output: true if BSDF was used, false if grid was used
 */
__device__ Vector3f sampleMIS(const Grid& grid, const Vector3f& normal, 
                               curandState* rng, float& weight, float bsdf_prob,
                               bool& used_bsdf) {
    const float BSDF_PROB = fmaxf(fminf(bsdf_prob, 0.99f), 0.01f);  // Clamp to avoid div by 0
    const float GRID_PROB = 1.0f - BSDF_PROB;
    
    float xi = curand_uniform(rng);
    Vector3f dir;
    float pdf_grid, pdf_bsdf;
    float mis_w;  // MIS weight (power heuristic)
    
    if (xi < BSDF_PROB) {
        // ===== BSDF SAMPLING =====
        used_bsdf = true;
        dir = sampleCosineHemisphere(normal, rng);
        float cos_theta = fmaxf(dot(dir, normal), 0.0f);
        pdf_bsdf = cos_theta / M_PI;
        pdf_grid = grid.computePDF(dir, normal);
        
        // MIS weight: how much to trust this sample vs if we had used grid
        // Power heuristic: pdf_bsdf² / (pdf_bsdf² + pdf_grid²)
        mis_w = misPowerHeuristic(pdf_bsdf, pdf_grid);
        
        // Importance sampling weight for BSDF sample:
        // f(x) * cos / pdf_bsdf, but f = albedo/π and we already multiplied albedo
        // So: (1/π) * cos / pdf_bsdf = (1/π) * cos / (cos/π) = 1
        // With MIS: weight = mis_w / BSDF_PROB (divide by selection probability)
        if (pdf_bsdf > 1e-6f) {
            weight = mis_w / BSDF_PROB;
        } else {
            weight = 0.0f;
        }
    } else {
        // ===== GRID SAMPLING =====
        used_bsdf = false;
        dir = grid.sample(normal, rng, pdf_grid);
        float cos_theta = fmaxf(dot(dir, normal), 0.0f);
        pdf_bsdf = cos_theta / M_PI;
        
        // MIS weight: how much to trust this sample vs if we had used BSDF
        mis_w = misPowerHeuristic(pdf_grid, pdf_bsdf);
        
        // Importance sampling weight for grid sample:
        // f(x) * cos / pdf_grid = (1/π) * cos / pdf_grid
        // With MIS: weight = mis_w * cos / (π * pdf_grid * GRID_PROB)
        if (pdf_grid > 1e-6f && cos_theta > 0.0f) {
            weight = mis_w * cos_theta / (M_PI * pdf_grid * GRID_PROB);
            weight = fminf(weight, 10.0f);  // Clamp to prevent fireflies
        } else {
            weight = 0.0f;
        }
    }
    
    return dir;
}

// Overload for backward compatibility (default bsdf_prob = 0.5)
__device__ Vector3f sampleMIS(const Grid& grid, const Vector3f& normal, 
                               curandState* rng, float& weight, float bsdf_prob = 0.5f) {
    bool unused;
    return sampleMIS(grid, normal, rng, weight, bsdf_prob, unused);
}

// ============================================================================
// PATH INTEGRATOR - Clean, readable, modular
// ============================================================================

/**
 * Main path tracing integrator
 * 
 * @param scene      The scene to render
 * @param ray        Initial ray from camera  
 * @param L          Output radiance (accumulated)
 * @param max_depth  Maximum path length
 * @param rng        Random number generator state
 * @param mode       Sampling mode (BSDF, Grid, or MIS)
 */
__device__ void integrator(const Scene* scene, Ray& ray, Vector3f& L, 
                           int max_depth, curandState* rng, SamplingMode mode) {
    
    Vector3f throughput(1.0f, 1.0f, 1.0f);  // Path throughput (β)
    Ray r = ray;
    
    for (int depth = 0; depth < max_depth; depth++) {
        
        // ===================== RAY INTERSECTION =====================
        SurfaceInteractionRecord si;
        if (!scene->intersect(r, 1e-4f, FLT_MAX, si)) {
            break;  // Miss - no contribution
        }
        
        // ===================== EMISSION =====================
        L += throughput * si.Le;
        
        // ===================== RUSSIAN ROULETTE =====================
        if (depth > 2) {
            float max_throughput = fmaxf(throughput.x(), fmaxf(throughput.y(), throughput.z()));
            float rr_prob = fminf(max_throughput, 0.95f);
            if (curand_uniform(rng) > rr_prob) break;
            throughput /= rr_prob;
        }
        
        // ===================== BSDF =====================
        throughput *= si.bsdf;
        
        // Early termination
        if (throughput.length() < 1e-5f) break;
        
        // ===================== NEXT DIRECTION =====================
        Vector3f normal = si.n;
        Vector3f shading_normal = dot(r.direction(), normal) < 0 ? normal : -normal;
        
        Vector3f next_dir;
        float weight = 1.0f;
        
        // ============ SAMPLING STRATEGY ============
        if (mode == SamplingMode::SAMPLING_BSDF) {
            // Pure cosine-weighted sampling
            next_dir = sampleCosineHemisphere(shading_normal, rng);
        }
        else {
            // Grid-guided sampling
            Grid grid;
            bool has_grid = initGridFromPrimitive(si, scene, grid);
            
            if (has_grid && grid.isValid()) {
                if (mode == SamplingMode::SAMPLING_MIS) {
                    // MIS: mix grid and BSDF sampling with configurable fraction
                    next_dir = sampleMIS(grid, shading_normal, rng, weight, scene->mis_bsdf_fraction);
                    throughput *= weight;
                } else {
                    // Pure grid sampling
                    // The grid stores incoming radiance (already weighted by geometry)
                    // For unbiased estimate: weight = cos(theta) / (pi * pdf)
                    float grid_pdf;
                    next_dir = grid.sample(shading_normal, rng, grid_pdf);
                    
                    float cos_theta = fmaxf(dot(next_dir, shading_normal), 0.0f);
                    
                    // For Lambertian: BRDF = albedo/pi, we already multiplied by albedo
                    // Full contribution: L * albedo * cos / pi
                    // With grid sampling (pdf = grid_pdf): L * albedo * cos / (pi * grid_pdf)
                    // Since we want to be unbiased w.r.t. cosine sampling:
                    weight = cos_theta / (M_PI * fmaxf(grid_pdf, 1e-6f));
                    weight = fminf(fmaxf(weight, 0.0f), 10.0f);  // Clamp to prevent fireflies
                    throughput *= weight;
                }
            } else {
                // Fallback to cosine sampling
                next_dir = sampleCosineHemisphere(shading_normal, rng);
            }
        }
        
        // ===================== SPAWN NEW RAY =====================
        r = Ray(si.p + shading_normal * 1e-4f, next_dir);
    }
}

// ============================================================================
// RENDER KERNELS
// ============================================================================

__global__ void render_init(int width, int height, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    curand_init(2023 + pixel_index, pixel_index, 0, &rand_state[pixel_index]);
}

// ============================================================================
// PROFILED INTEGRATOR - Tracks cycle counts for performance analysis
// ============================================================================

#if ENABLE_KERNEL_PROFILING
__device__ void integrator_profiled(const Scene* scene, Ray& ray, Vector3f& L, 
                                     int max_depth, curandState* rng, SamplingMode mode,
                                     KernelProfileData* profile) {
    
    Vector3f throughput(1.0f, 1.0f, 1.0f);
    Ray r = ray;
    
    for (int depth = 0; depth < max_depth; depth++) {
        
        // ===================== RAY INTERSECTION (PROFILED) =====================
        long long t0 = clock64();
        SurfaceInteractionRecord si;
        bool hit = scene->intersect(r, 1e-4f, FLT_MAX, si);
        long long t1 = clock64();
        atomicAdd(&profile->intersection_cycles, (unsigned long long)(t1 - t0));
        
        if (!hit) break;
        
        L += throughput * si.Le;
        
        // Russian roulette
        if (depth > 2) {
            float max_throughput = fmaxf(throughput.x(), fmaxf(throughput.y(), throughput.z()));
            float rr_prob = fminf(max_throughput, 0.95f);
            if (curand_uniform(rng) > rr_prob) break;
            throughput /= rr_prob;
        }
        
        // ===================== SHADING (PROFILED) =====================
        long long t2 = clock64();
        throughput *= si.bsdf;
        long long t3 = clock64();
        atomicAdd(&profile->shading_cycles, (unsigned long long)(t3 - t2));
        
        if (throughput.length() < 1e-5f) break;
        
        Vector3f normal = si.n;
        Vector3f shading_normal = dot(r.direction(), normal) < 0 ? normal : -normal;
        
        Vector3f next_dir;
        float weight = 1.0f;
        
        // ===================== SAMPLING (PROFILED) =====================
        long long t4 = clock64();
        
        if (mode == SamplingMode::SAMPLING_BSDF) {
            next_dir = sampleCosineHemisphere(shading_normal, rng);
        }
        else {
            // ============ GRID INIT (PROFILED) ============
            long long tg0 = clock64();
            Grid grid;
            bool has_grid = initGridFromPrimitive(si, scene, grid);
            long long tg1 = clock64();
            atomicAdd(&profile->grid_init_cycles, (unsigned long long)(tg1 - tg0));
            
            if (has_grid && grid.isValid()) {
                atomicAdd(&profile->grid_samples, 1ULL);
                
                if (mode == SamplingMode::SAMPLING_MIS) {
                    next_dir = sampleMIS(grid, shading_normal, rng, weight, scene->mis_bsdf_fraction);
                    throughput *= weight;
                } else {
                    float grid_pdf;
                    next_dir = grid.sample(shading_normal, rng, grid_pdf);
                    float cos_theta = fmaxf(dot(next_dir, shading_normal), 0.0f);
                    weight = cos_theta / (M_PI * fmaxf(grid_pdf, 1e-6f));
                    weight = fminf(fmaxf(weight, 0.0f), 10.0f);
                    throughput *= weight;
                }
            } else {
                next_dir = sampleCosineHemisphere(shading_normal, rng);
            }
        }
        
        long long t5 = clock64();
        atomicAdd(&profile->sampling_cycles, (unsigned long long)(t5 - t4));
        atomicAdd(&profile->total_samples, 1ULL);
        
        r = Ray(si.p + shading_normal * 1e-4f, next_dir);
    }
}
#endif

__global__ void render(unsigned char* image, Sensor* cam, Scene* scene, 
                       curandState* rand_state, int spp, SamplingMode mode) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= cam->image_width || y >= cam->image_height) return;
    
    int pixel_index = y * cam->image_width + x;
    curandState* local_rng = &rand_state[pixel_index];
    
    Vector3f color(0.0f, 0.0f, 0.0f);
    
    for (int s = 0; s < spp; s++) {
        float u = (x + curand_uniform(local_rng)) / float(cam->image_width);
        float v = (y + curand_uniform(local_rng)) / float(cam->image_height);
        
        Ray ray = cam->get_ray(u, v);
        Vector3f sample_color(0.0f, 0.0f, 0.0f);
        integrator(scene, ray, sample_color, 5, local_rng, mode);
        color += sample_color;
    }
    
    color /= float(spp);
    
    // Tone mapping (Reinhard)
    color = color / (color + Vector3f(1.0f, 1.0f, 1.0f));
    
    // Gamma correction
    const float gamma = 1.0f / 2.2f;
    color.e[0] = powf(color.e[0], gamma);
    color.e[1] = powf(color.e[1], gamma);
    color.e[2] = powf(color.e[2], gamma);
    
    int idx = (y * cam->image_width + x) * 3;
    image[idx + 0] = (unsigned char)(255.99f * fminf(color.r(), 1.0f));
    image[idx + 1] = (unsigned char)(255.99f * fminf(color.g(), 1.0f));
    image[idx + 2] = (unsigned char)(255.99f * fminf(color.b(), 1.0f));
}

// ============================================================================
// PROFILED RENDER KERNEL
// ============================================================================

#if ENABLE_KERNEL_PROFILING
__global__ void render_profiled(unsigned char* image, Sensor* cam, Scene* scene, 
                                 curandState* rand_state, int spp, SamplingMode mode,
                                 KernelProfileData* profile) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= cam->image_width || y >= cam->image_height) return;
    
    int pixel_index = y * cam->image_width + x;
    curandState* local_rng = &rand_state[pixel_index];
    
    Vector3f color(0.0f, 0.0f, 0.0f);
    
    for (int s = 0; s < spp; s++) {
        float u = (x + curand_uniform(local_rng)) / float(cam->image_width);
        float v = (y + curand_uniform(local_rng)) / float(cam->image_height);
        
        Ray ray = cam->get_ray(u, v);
        Vector3f sample_color(0.0f, 0.0f, 0.0f);
        integrator_profiled(scene, ray, sample_color, 5, local_rng, mode, profile);
        color += sample_color;
    }
    
    color /= float(spp);
    
    // Tone mapping (Reinhard)
    color = color / (color + Vector3f(1.0f, 1.0f, 1.0f));
    
    // Gamma correction
    const float gamma = 1.0f / 2.2f;
    color.e[0] = powf(color.e[0], gamma);
    color.e[1] = powf(color.e[1], gamma);
    color.e[2] = powf(color.e[2], gamma);
    
    int idx = (y * cam->image_width + x) * 3;
    image[idx + 0] = (unsigned char)(255.99f * fminf(color.r(), 1.0f));
    image[idx + 1] = (unsigned char)(255.99f * fminf(color.g(), 1.0f));
    image[idx + 2] = (unsigned char)(255.99f * fminf(color.b(), 1.0f));
}
#endif

// ============================================================================
// RADIOSITY VISUALIZATION KERNEL
// ============================================================================

__global__ void render_radiosity(unsigned char* image, Sensor* cam, Scene* scene,
                                  curandState* rand_state, int spp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= cam->image_width || y >= cam->image_height) return;
    
    int pixel_index = y * cam->image_width + x;
    curandState local_rng = rand_state[pixel_index];
    
    Vector3f color(0.0f, 0.0f, 0.0f);
    
    for (int s = 0; s < spp; s++) {
        float u = (x + curand_uniform(&local_rng)) / float(cam->image_width);
        float v = (y + curand_uniform(&local_rng)) / float(cam->image_height);
        
        Ray ray = cam->get_ray(u, v);
        SurfaceInteractionRecord si;
        
        if (scene->intersect(ray, 1e-4f, FLT_MAX, si)) {
            color += si.Le;
            if (si.prim_ptr != nullptr) {
                color += si.prim_ptr->getRadiosity();
            } else {
                color += si.radiosity;
            }
        }
    }
    
    color /= float(spp);
    
    // Gamma correction
    color = Vector3f(
        sqrtf(fminf(color.x(), 1.0f)),
        sqrtf(fminf(color.y(), 1.0f)),
        sqrtf(fminf(color.z(), 1.0f))
    );
    
    int idx = 3 * pixel_index;
    image[idx + 0] = (unsigned char)(255.99f * color.x());
    image[idx + 1] = (unsigned char)(255.99f * color.y());
    image[idx + 2] = (unsigned char)(255.99f * color.z());
    
    rand_state[pixel_index] = local_rng;
}

#endif // INTEGRATOR_H
