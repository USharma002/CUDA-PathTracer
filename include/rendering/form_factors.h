#ifndef FORM_FACTORS_H
#define FORM_FACTORS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "core/vector.h"
#include "rendering/triangle.h"
#include "rendering/quad.h"
#include "rendering/bvh.h"
#include "core/ray.h"
#include "rendering/render_config.h"
#include "rendering/grid_filter.h"
#include "utils/optix_logger.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ==================== FORM FACTOR COMPUTATION STATS ====================

struct FormFactorStats {
    int total_pairs;
    int total_samples;
    long long visibility_tests;
    long long rays_cast;
    double compute_time_ms;
    
    void reset() {
        total_pairs = 0;
        total_samples = 0;
        visibility_tests = 0;
        rays_cast = 0;
        compute_time_ms = 0.0;
    }
    
    void log() {
        auto& logger = OptixLogger::getInstance();
        std::stringstream ss;
        ss << "Form Factor Statistics:";
        logger.info("FormFactors", ss.str());
        
        ss.str(""); ss << "  Total pairs: " << total_pairs;
        logger.info("FormFactors", ss.str());
        
        ss.str(""); ss << "  Total samples: " << total_samples;
        logger.info("FormFactors", ss.str());
        
        ss.str(""); ss << "  Visibility tests: " << visibility_tests;
        logger.info("FormFactors", ss.str());
        
        ss.str(""); ss << "  Rays cast: " << rays_cast;
        logger.info("FormFactors", ss.str());
        
        ss.str(""); ss << "  Compute time: " << std::fixed << std::setprecision(2) << compute_time_ms << " ms";
        logger.info("FormFactors", ss.str());
        
        if (compute_time_ms > 0 && rays_cast > 0) {
            double mrays_per_sec = (rays_cast / 1000000.0) / (compute_time_ms / 1000.0);
            ss.str(""); ss << "  Performance: " << std::fixed << std::setprecision(2) << mrays_per_sec << " MRays/sec";
            logger.info("FormFactors", ss.str());
        }
    }
};

// Global stats (can be accessed from host)
static FormFactorStats g_form_factor_stats;

// ==================== INITIALIZATION ====================

__global__ void initialize_directional_grids(Primitive* primitives, int num_primitives) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_primitives) return;
    
    float* grid = primitives[i].getGrid();
    Vector3f* rad_grid = primitives[i].getRadiosityGrid();
    
    for (int idx = 0; idx < GRID_RESOLUTION * GRID_RESOLUTION; idx++) {
        grid[idx] = 0.0f;
        rad_grid[idx] = Vector3f(0.0f, 0.0f, 0.0f);
    }
}

__global__ void formfactor_rand_init(int num_pairs, curandState* rand_states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_pairs) return;
    curand_init(12345 + idx, idx, 0, &rand_states[idx]);
}

// ==================== UTILITY FUNCTIONS ====================

// Build orthonormal frame from normal (Frisvad's method)
__device__ void build_frame(const Vector3f& n, Vector3f& t, Vector3f& b) {
    if (n.z() < -0.9999999f) {
        t = Vector3f(0.0f, -1.0f, 0.0f);
        b = Vector3f(-1.0f, 0.0f, 0.0f);
        return;
    }
    float a = 1.0f / (1.0f + n.z());
    float c = -n.x() * n.y() * a;
    t = Vector3f(1.0f - n.x() * n.x() * a, c, -n.x());
    b = Vector3f(c, 1.0f - n.y() * n.y() * a, -n.y());
}

// Convert world-space direction to LOCAL-space grid indices (relative to surface normal)
// This ensures the grid is parameterized by local spherical coordinates
__device__ void direction_to_grid_indices_local(const Vector3f& world_dir, const Vector3f& normal,
                                                 int& grid_theta, int& grid_phi) {
    // Transform world direction to local frame (normal = local z-axis)
    Vector3f tangent, bitangent;
    build_frame(normal, tangent, bitangent);
    
    float lx = dot(world_dir, tangent);
    float ly = dot(world_dir, bitangent);
    float lz = dot(world_dir, normal);
    
    // Compute local spherical coordinates
    float r = sqrtf(lx*lx + ly*ly + lz*lz);
    float theta = (r > 0.0f) ? acosf(fminf(lz / r, 1.0f)) : 0.0f;  // angle from normal
    float phi = atan2f(ly, lx);
    if (phi < 0.0f) phi += 2.0f * M_PI;
    
    // Map to grid indices - theta covers [0, pi], first half is upper hemisphere
    grid_theta = (int)fminf((theta / M_PI) * GRID_RESOLUTION, GRID_RESOLUTION - 1);
    grid_phi = (int)fminf((phi / (2.0f * M_PI)) * GRID_RESOLUTION, GRID_RESOLUTION - 1);
    grid_theta = max(0, min(grid_theta, GRID_RESOLUTION - 1));
    grid_phi = max(0, min(grid_phi, GRID_RESOLUTION - 1));
}

// Legacy world-space version (for backward compatibility if needed)
__device__ void direction_to_grid_indices(const Vector3f& dir, int& grid_theta, int& grid_phi) {
    float r = dir.length();
    float theta = (r > 0.0f) ? acosf(dir.z() / r) : 0.0f;
    float phi = atan2f(dir.y(), dir.x());
    if (phi < 0.0f) phi += 2.0f * M_PI;
    
    grid_theta = fminf((theta / M_PI) * (GRID_RESOLUTION - 1), GRID_RESOLUTION - 1);
    grid_phi = fminf((phi / (2.0f * M_PI)) * (GRID_RESOLUTION - 1), GRID_RESOLUTION - 1);
}

// ==================== OPTIMIZED VISIBILITY TESTING ====================

// Fast any-hit test - returns true if ray is blocked (early termination)
__device__ bool visibility_test_anyhit(const Ray& r, float max_dist,
                                        const BVHNode* bvh_nodes, const int* bvh_indices,
                                        const Primitive* primitives, int num_primitives,
                                        int source_idx, int target_idx) {
    const float EPSILON = 1e-5f;
    
    if (bvh_nodes != nullptr) {
        int stack[32];  // Reduced stack size for speed
        int stack_ptr = 0;
        stack[stack_ptr++] = 0;
        
        // Precompute inverse direction
        Vector3f inv_dir(1.0f / (fabsf(r.direction().x()) > 1e-8f ? r.direction().x() : 1e-8f),
                         1.0f / (fabsf(r.direction().y()) > 1e-8f ? r.direction().y() : 1e-8f),
                         1.0f / (fabsf(r.direction().z()) > 1e-8f ? r.direction().z() : 1e-8f));
        
        while (stack_ptr > 0) {
            int node_idx = stack[--stack_ptr];
            const BVHNode& node = bvh_nodes[node_idx];
            
            // Fast slab-based AABB test
            float t1 = (node.bbox.min[0] - r.origin().x()) * inv_dir.x();
            float t2 = (node.bbox.max[0] - r.origin().x()) * inv_dir.x();
            float tmin = fminf(t1, t2);
            float tmax = fmaxf(t1, t2);
            
            t1 = (node.bbox.min[1] - r.origin().y()) * inv_dir.y();
            t2 = (node.bbox.max[1] - r.origin().y()) * inv_dir.y();
            tmin = fmaxf(tmin, fminf(t1, t2));
            tmax = fminf(tmax, fmaxf(t1, t2));
            
            t1 = (node.bbox.min[2] - r.origin().z()) * inv_dir.z();
            t2 = (node.bbox.max[2] - r.origin().z()) * inv_dir.z();
            tmin = fmaxf(tmin, fminf(t1, t2));
            tmax = fminf(tmax, fmaxf(t1, t2));
            
            if (tmax < EPSILON || tmin > max_dist || tmin > tmax) continue;
            
            if (node.isLeaf()) {
                for (int i = 0; i < node.prim_count; i++) {
                    int prim_idx = bvh_indices[node.left_child + i];
                    if (prim_idx == source_idx || prim_idx == target_idx) continue;
                    
                    SurfaceInteractionRecord temp;
                    if (primitives[prim_idx].intersect(r, EPSILON, max_dist, temp)) {
                        return true;  // Blocked! Early exit
                    }
                }
            } else if (stack_ptr < 30) {
                stack[stack_ptr++] = node.left_child;
                stack[stack_ptr++] = node.right_child;
            }
        }
        return false;  // Not blocked
    } else {
        // Linear search fallback
        for (int i = 0; i < num_primitives; ++i) {
            if (i == source_idx || i == target_idx) continue;
            SurfaceInteractionRecord temp;
            if (primitives[i].intersect(r, EPSILON, max_dist, temp)) {
                return true;
            }
        }
        return false;
    }
}

// ==================== FORM FACTOR KERNELS ====================

// OPTIMIZED Monte Carlo form factor computation
// Key optimizations:
// 1. Early culling of back-facing pairs
// 2. Adaptive sampling based on distance
// 3. Fast any-hit visibility test
// 4. Reduced atomic operations

__global__ void calculate_form_factors_mc_kernel(float* form_factors, Primitive* primitives,
                                                  int num_primitives, int n_samples,
                                                  curandState* rand_states, BVHNode* bvh_nodes,
                                                  int* bvh_indices) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= num_primitives || j >= num_primitives) return;
    
    int ff_idx = i * num_primitives + j;
    
    if (i == j) {
        form_factors[ff_idx] = 0.0f;
        return;
    }
    
    const Primitive& prim_i = primitives[i];
    const Primitive& prim_j = primitives[j];
    Vector3f center_i = prim_i.centroid();
    Vector3f center_j = prim_j.centroid();
    Vector3f normal_i = prim_i.getNormal();
    Vector3f normal_j = prim_j.getNormal();
    
    // ===== EARLY CULLING =====
    // Check if patches face each other using centroids
    Vector3f dir_ij = center_j - center_i;
    float dist_sq = dir_ij.x()*dir_ij.x() + dir_ij.y()*dir_ij.y() + dir_ij.z()*dir_ij.z();
    float dist = sqrtf(dist_sq);
    
    if (dist < 1e-6f) {
        form_factors[ff_idx] = 0.0f;
        return;
    }
    
    Vector3f dir_norm = dir_ij / dist;
    float cos_i_approx = dot(normal_i, dir_norm);
    float cos_j_approx = -dot(normal_j, dir_norm);
    
    // Skip if patches definitely don't face each other
    if (cos_i_approx <= 0.0f || cos_j_approx <= 0.0f) {
        form_factors[ff_idx] = 0.0f;
        return;
    }
    
    // ===== ADAPTIVE SAMPLING =====
    // Use fewer samples for distant/small form factors
    float approx_ff = (cos_i_approx * cos_j_approx * prim_j.getArea()) / (M_PI * dist_sq);
    int actual_samples = n_samples;
    if (approx_ff < 0.001f) actual_samples = max(1, n_samples / 4);
    else if (approx_ff < 0.01f) actual_samples = max(2, n_samples / 2);
    
    curandState local_state = rand_states[ff_idx];
    
    float visibility_sum = 0.0f;
    float cos_i_sum = 0.0f, cos_j_sum = 0.0f, dist_sum = 0.0f;
    int valid_samples = 0;
    
    // Local accumulators for grid updates (reduce atomics)
    float local_grid[GRID_SIZE] = {0.0f};
    Vector3f local_rad_grid[GRID_SIZE];
    for (int k = 0; k < GRID_SIZE; k++) local_rad_grid[k] = Vector3f(0,0,0);
    
    for (int s = 0; s < actual_samples; ++s) {
        float r1 = curand_uniform(&local_state);
        float r2 = curand_uniform(&local_state);
        Vector3f p_i = prim_i.sampleUniform(r1, r2);
        
        r1 = curand_uniform(&local_state);
        r2 = curand_uniform(&local_state);
        Vector3f p_j = prim_j.sampleUniform(r1, r2);
        
        Vector3f sample_dir = p_j - p_i;
        float r = sample_dir.length();
        if (r < 1e-6f) continue;
        
        sample_dir = sample_dir / r;
        
        float cos_theta_i = dot(normal_i, sample_dir);
        float cos_theta_j = -dot(normal_j, sample_dir);
        if (cos_theta_i <= 0.0f || cos_theta_j <= 0.0f) continue;
        
        // Fast any-hit visibility test
        Ray shadow_ray(p_i + normal_i * 1e-4f, sample_dir);
        bool blocked = visibility_test_anyhit(shadow_ray, r - 2e-4f, bvh_nodes, bvh_indices,
                                               primitives, num_primitives, i, j);
        
        if (!blocked) {
            visibility_sum += 1.0f;
            cos_i_sum += cos_theta_i;
            cos_j_sum += cos_theta_j;
            dist_sum += r;
            valid_samples++;
            
            // Accumulate locally first
            int grid_theta, grid_phi;
            direction_to_grid_indices_local(sample_dir, normal_i, grid_theta, grid_phi);
            int grid_idx = grid_theta * GRID_RESOLUTION + grid_phi;
            
            local_grid[grid_idx] += 1.0f;
            
            float geometric_weight = (cos_theta_i * cos_theta_j) / (r * r);
            Vector3f contrib = prim_j.getRadiosity() * geometric_weight * prim_j.getArea();
            local_rad_grid[grid_idx] += contrib;
        }
    }
    
    rand_states[ff_idx] = local_state;
    
    // Single atomic batch update for grids (much faster than per-sample atomics)
    float* grid = primitives[i].getGrid();
    Vector3f* rad_grid = primitives[i].getRadiosityGrid();
    for (int k = 0; k < GRID_SIZE; k++) {
        if (local_grid[k] > 0.0f) {
            atomicAdd(&grid[k], local_grid[k]);
            atomicAdd(&rad_grid[k].e[0], local_rad_grid[k].x());
            atomicAdd(&rad_grid[k].e[1], local_rad_grid[k].y());
            atomicAdd(&rad_grid[k].e[2], local_rad_grid[k].z());
        }
    }
    
    if (valid_samples > 0) {
        float avg_cos_i = cos_i_sum / valid_samples;
        float avg_cos_j = cos_j_sum / valid_samples;
        float avg_dist = dist_sum / valid_samples;
        float visibility_fraction = visibility_sum / actual_samples;
        
        float F_ij = visibility_fraction * (avg_cos_i * avg_cos_j * prim_j.getArea()) / 
                     (M_PI * avg_dist * avg_dist);
        form_factors[ff_idx] = fmaxf(0.0f, fminf(F_ij, 1.0f));
    } else {
        form_factors[ff_idx] = 0.0f;
    }
}

__global__ void calculate_form_factors_kernel(float* form_factors, Primitive* primitives, 
                                             int num_primitives, BVHNode* bvh_nodes,
                                             int* bvh_indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= num_primitives || j >= num_primitives) return;
    
    int ff_idx = i * num_primitives + j;
    
    if (i == j) {
        form_factors[ff_idx] = 0.0f;
        return;
    }
    
    const Primitive& prim_i = primitives[i];
    const Primitive& prim_j = primitives[j];
    
    Vector3f vec_ij = prim_j.centroid() - prim_i.centroid();
    float r = vec_ij.length();
    if (r < 1e-6f) {
        form_factors[ff_idx] = 0.0f;
        return;
    }
    
    Vector3f dir_ij = vec_ij / r;
    Vector3f normal_i = prim_i.getNormal();
    Vector3f normal_j = prim_j.getNormal();
    
    float cos_theta_i = dot(normal_i, dir_ij);
    float cos_theta_j = dot(normal_j, -dir_ij);
    
    // Early exit if patches don't face each other
    if (cos_theta_i <= 0.0f || cos_theta_j <= 0.0f) {
        form_factors[ff_idx] = 0.0f;
        return;
    }
    
    // Use fast any-hit visibility test
    Ray visibility_ray(prim_i.centroid() + normal_i * 1e-4f, dir_ij);
    bool blocked = visibility_test_anyhit(visibility_ray, r - 2e-4f, bvh_nodes, bvh_indices,
                                           primitives, num_primitives, i, j);
    
    if (blocked) {
        form_factors[ff_idx] = 0.0f;
        return;
    }
    
    float ff = (cos_theta_i * cos_theta_j * prim_j.getArea()) / (M_PI * r * r);
    form_factors[ff_idx] = fmaxf(0.0f, ff);
}

// ==================== RADIOSITY ====================

__global__ void update_radiosity_grid(Primitive* primitives, float* form_factors, int num_primitives) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_primitives) return;
    
    Vector3f center_i = primitives[i].centroid();
    Vector3f normal_i = primitives[i].getNormal();  // Get surface normal
    Vector3f* rad_grid = primitives[i].getRadiosityGrid();
    
    for (int idx = 0; idx < GRID_RESOLUTION * GRID_RESOLUTION; idx++) {
        rad_grid[idx] = Vector3f(0.0f, 0.0f, 0.0f);
    }
    
    for (int j = 0; j < num_primitives; ++j) {
        if (i == j) continue;
        
        float F_ij = form_factors[i * num_primitives + j];
        if (F_ij <= 0.0f) continue;
        
        Vector3f dir_ij = primitives[j].centroid() - center_i;
        float r = dir_ij.length();
        if (r < 1e-6f) continue;
        
        dir_ij = dir_ij / r;
        
        int grid_theta, grid_phi;
        // Use LOCAL-space indexing relative to surface normal
        direction_to_grid_indices_local(dir_ij, normal_i, grid_theta, grid_phi);
        int grid_idx = grid_theta * GRID_RESOLUTION + grid_phi;
        
        Vector3f contribution = primitives[j].getRadiosity() * F_ij;
        atomicAdd(&rad_grid[grid_idx].e[0], contribution.x());
        atomicAdd(&rad_grid[grid_idx].e[1], contribution.y());
        atomicAdd(&rad_grid[grid_idx].e[2], contribution.z());
    }
}

__global__ void radiosity_iteration_kernel(Primitive* primitives, float* form_factors, int num_primitives) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_primitives) return;
    
    Vector3f incident_rad(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < num_primitives; ++j) {
        if (i != j) {
            float F_ij = form_factors[i * num_primitives + j];
            if (F_ij > 0.0f) {
                incident_rad += primitives[j].getUnshotRad() * F_ij;
            }
        }
    }
    
    Vector3f bsdf = primitives[i].getBSDF();
    Vector3f reflected = Vector3f(
        fminf(bsdf.x() * incident_rad.x(), incident_rad.x()),
        fminf(bsdf.y() * incident_rad.y(), incident_rad.y()),
        fminf(bsdf.z() * incident_rad.z(), incident_rad.z())
    );
    
    primitives[i].setRadiosity(primitives[i].getRadiosity() + reflected);
    primitives[i].setUnshotRad(reflected);
}

// ==================== SUBDIVISION ====================

Vector3f calculate_normal(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2) {
    return unit_vector(cross(v1 - v0, v2 - v0));
}

std::vector<Triangle> subdivide_triangle(const Triangle& tri) {
    Vector3f m0 = (tri.v0 + tri.v1) * 0.5f;
    Vector3f m1 = (tri.v1 + tri.v2) * 0.5f;
    Vector3f m2 = (tri.v2 + tri.v0) * 0.5f;
    
    std::vector<Triangle> result;
    Triangle subtris[4] = {
        Triangle(tri.v0, m0, m2, tri.bsdf),
        Triangle(m0, tri.v1, m1, tri.bsdf),
        Triangle(m1, tri.v2, m2, tri.bsdf),
        Triangle(m0, m1, m2, tri.bsdf)
    };
    
    for (int i = 0; i < 4; i++) {
        subtris[i].normal = calculate_normal(subtris[i].v0, subtris[i].v1, subtris[i].v2);
        subtris[i].Le = tri.Le;
        result.push_back(subtris[i]);
    }
    
    return result;
}

std::vector<Quad> subdivide_quad(const Quad& quad) {
    Vector3f m01 = (quad.v00 + quad.v10) * 0.5f;
    Vector3f m12 = (quad.v10 + quad.v11) * 0.5f;
    Vector3f m23 = (quad.v11 + quad.v01) * 0.5f;
    Vector3f m30 = (quad.v01 + quad.v00) * 0.5f;
    Vector3f center = (quad.v00 + quad.v10 + quad.v11 + quad.v01) * 0.25f;
    
    std::vector<Quad> result;
    Quad subquads[4] = {
        Quad(quad.v00, m01, center, m30, quad.bsdf),
        Quad(m01, quad.v10, m12, center, quad.bsdf),
        Quad(center, m12, quad.v11, m23, quad.bsdf),
        Quad(m30, center, m23, quad.v01, quad.bsdf)
    };
    
    for (int i = 0; i < 4; i++) {
        subquads[i].Le = quad.Le;
        result.push_back(subquads[i]);
    }
    
    return result;
}

std::vector<Primitive> subdivide_primitives(const std::vector<Primitive>& prims, int num_subdivisions) {
    auto& log = OptixLogger::getInstance();
    
    if (num_subdivisions == 0) {
        log.debug("FormFactors", "No subdivision requested");
        return prims;
    }
    
    log.info("FormFactors", "========== Primitive Subdivision ==========");
    std::stringstream ss;
    ss << "Input primitives: " << prims.size() << ", Subdivision levels: " << num_subdivisions;
    log.info("FormFactors", ss.str());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<Primitive> current_list = prims;
    
    for (int iter = 0; iter < num_subdivisions; ++iter) {
        std::vector<Primitive> next_list;
        int tri_count = 0, quad_count = 0;
        
        for (const auto& prim : current_list) {
            if (prim.type == PRIM_TRIANGLE) {
                for (const auto& tri : subdivide_triangle(prim.tri)) {
                    next_list.push_back(Primitive(tri));
                    tri_count++;
                }
            } else {
                for (const auto& q : subdivide_quad(prim.quad)) {
                    next_list.push_back(Primitive(q));
                    quad_count++;
                }
            }
        }
        
        ss.str("");
        ss << "  Level " << (iter + 1) << ": " << current_list.size() << " -> " << next_list.size() 
           << " primitives (" << tri_count << " triangles, " << quad_count << " quads)";
        log.debug("FormFactors", ss.str());
        
        current_list = next_list;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    ss.str("");
    ss << "Subdivision complete: " << prims.size() << " -> " << current_list.size() 
       << " primitives in " << std::fixed << std::setprecision(2) << elapsed_ms << " ms";
    log.info("FormFactors", ss.str());
    log.info("FormFactors", "============================================");
    
    std::cout << "Subdivision: " << prims.size() << " -> " << current_list.size() << " primitives" << std::endl;
    return current_list;
}

__global__ void store_radiosity_history_kernel(Primitive* primitives, int num_prims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_prims) return;
    
    primitives[idx].pushRadiosityHistory(primitives[idx].getRadiosity());
}


#endif // FORM_FACTORS_H
