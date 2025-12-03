#ifndef FORM_FACTORS_H
#define FORM_FACTORS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "vector.h"
#include "triangle.h"
#include "quad.h"
#include "bvh.h"
#include "ray.h"
#include "render_config.h"
#include "grid_filter.h"  // Bilateral filtering for radiosity grids (can safely remove this line)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

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

__device__ void direction_to_grid_indices(const Vector3f& dir, int& grid_theta, int& grid_phi) {
    float r = dir.length();
    float theta = (r > 0.0f) ? acosf(dir.z() / r) : 0.0f;
    float phi = atan2f(dir.y(), dir.x());
    if (phi < 0.0f) phi += 2.0f * M_PI;
    
    grid_theta = fminf((theta / M_PI) * (GRID_RESOLUTION - 1), GRID_RESOLUTION - 1);
    grid_phi = fminf((phi / (2.0f * M_PI)) * (GRID_RESOLUTION - 1), GRID_RESOLUTION - 1);
}

// ==================== VISIBILITY TESTING ====================

__device__ bool visibility_test(const Ray& r, float max_dist,
                               const BVHNode* bvh_nodes, const int* bvh_indices,
                               const Primitive* primitives, int num_primitives,
                               int target_idx, int source_idx) {
    const float EPSILON = 1e-6f;
    const float TOLERANCE = 1e-4f;
    
    float closest_t = max_dist + 1.0f;
    int closest_idx = -1;
    
    if (bvh_nodes != nullptr) {
        // BVH traversal
        int stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0;
        
        while (stack_ptr > 0) {
            int node_idx = stack[--stack_ptr];
            const BVHNode& node = bvh_nodes[node_idx];
            
            if (!node.bbox.intersect(r, EPSILON, closest_t)) continue;
            
            if (node.isLeaf()) {
                for (int i = 0; i < node.prim_count; i++) {
                    int prim_idx = bvh_indices[node.left_child + i];
                    if (prim_idx == source_idx) continue;
                    
                    SurfaceInteractionRecord temp;
                    if (primitives[prim_idx].intersect(r, EPSILON, closest_t, temp)) {
                        if (temp.t < closest_t) {
                            closest_t = temp.t;
                            closest_idx = prim_idx;
                        }
                    }
                }
            } else if (stack_ptr < 62) {
                stack[stack_ptr++] = node.left_child;
                stack[stack_ptr++] = node.right_child;
            }
        }
    } else {
        // Linear search fallback
        for (int i = 0; i < num_primitives; ++i) {
            if (i == source_idx) continue;
            
            SurfaceInteractionRecord temp;
            if (primitives[i].intersect(r, EPSILON, closest_t, temp)) {
                if (temp.t < closest_t) {
                    closest_t = temp.t;
                    closest_idx = i;
                }
            }
        }
    }
    
    return (closest_idx == target_idx && 
            fabsf(closest_t - max_dist) < max_dist * TOLERANCE + EPSILON);
}

// ==================== FORM FACTOR KERNELS ====================

// Generate random points on Patch i and Patch j, test visibility, accumulate form factor
// On an average will give area-weighted form factors using Monte Carlo integration
// Also update the directional grid, have to fix later

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
    
    curandState local_state = rand_states[ff_idx];
    
    const Primitive& prim_i = primitives[i];
    const Primitive& prim_j = primitives[j];
    Vector3f normal_i = prim_i.getNormal();
    Vector3f normal_j = prim_j.getNormal();
    
    float visibility_sum = 0.0f, cos_i_sum = 0.0f, cos_j_sum = 0.0f, dist_sum = 0.0f;
    int valid_samples = 0;
    
    for (int s = 0; s < n_samples; ++s) {
        float r1 = curand_uniform(&local_state);
        float r2 = curand_uniform(&local_state);
        Vector3f p_i = prim_i.sampleUniform(r1, r2);
        
        r1 = curand_uniform(&local_state);
        r2 = curand_uniform(&local_state);
        Vector3f p_j = prim_j.sampleUniform(r1, r2);
        
        Vector3f dir_ij = p_j - p_i;
        float r = dir_ij.length();
        if (r < 1e-6f) continue;
        
        dir_ij = dir_ij / r;
        
        float cos_theta_i = dot(normal_i, dir_ij);
        float cos_theta_j = -dot(normal_j, dir_ij);
        if (cos_theta_i <= 0.0f || cos_theta_j <= 0.0f) continue;
        
        Ray shadow_ray(p_i + normal_i * 1e-4f, dir_ij);
        
        if (visibility_test(shadow_ray, r - 2e-4f, bvh_nodes, bvh_indices,
                          primitives, num_primitives, j, i)) {
            visibility_sum += 1.0f;
            cos_i_sum += cos_theta_i;
            cos_j_sum += cos_theta_j;
            dist_sum += r;
            valid_samples++;
            
            // Update directional grid
            int grid_theta, grid_phi;
            direction_to_grid_indices(dir_ij, grid_theta, grid_phi);
            int grid_idx = grid_theta * GRID_RESOLUTION + grid_phi;
            
            atomicAdd(&primitives[i].getGrid()[grid_idx], 1.0f);
            
            float geometric_weight = (cos_theta_i * cos_theta_j) / (r * r);
            Vector3f sample_radiosity = prim_j.getRadiosity() * geometric_weight * prim_j.getArea();
            
            Vector3f* rad_grid = primitives[i].getRadiosityGrid();
            atomicAdd(&rad_grid[grid_idx].e[0], sample_radiosity.x());
            atomicAdd(&rad_grid[grid_idx].e[1], sample_radiosity.y());
            atomicAdd(&rad_grid[grid_idx].e[2], sample_radiosity.z());
        }
    }
    
    rand_states[ff_idx] = local_state;
    
    if (valid_samples > 0) {
        float avg_cos_i = cos_i_sum / valid_samples;
        float avg_cos_j = cos_j_sum / valid_samples;
        float avg_dist = dist_sum / valid_samples;
        float visibility_fraction = visibility_sum / n_samples;
        
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
    
    float cos_theta_i = fmaxf(0.0f, dot(normal_i, dir_ij));
    float cos_theta_j = fmaxf(0.0f, dot(normal_j, -dir_ij));
    
    if (cos_theta_i == 0.0f || cos_theta_j == 0.0f) {
        form_factors[ff_idx] = 0.0f;
        return;
    }
    
    Ray visibility_ray(prim_i.centroid() + normal_i * 1e-4f, dir_ij);
    
    if (!visibility_test(visibility_ray, r, bvh_nodes, bvh_indices,
                        primitives, num_primitives, j, i)) {
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
        direction_to_grid_indices(dir_ij, grid_theta, grid_phi);
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
    if (num_subdivisions == 0) return prims;
    
    std::vector<Primitive> current_list = prims;
    
    for (int iter = 0; iter < num_subdivisions; ++iter) {
        std::vector<Primitive> next_list;
        for (const auto& prim : current_list) {
            if (prim.type == PRIM_TRIANGLE) {
                for (const auto& tri : subdivide_triangle(prim.tri)) {
                    next_list.push_back(Primitive(tri));
                }
            } else {
                for (const auto& q : subdivide_quad(prim.quad)) {
                    next_list.push_back(Primitive(q));
                }
            }
        }
        current_list = next_list;
    }
    
    std::cout << "Subdivision: " << prims.size() << " -> " << current_list.size() << " primitives" << std::endl;
    return current_list;
}

__global__ void store_radiosity_history_kernel(Primitive* primitives, int num_prims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_prims) return;
    
    primitives[idx].pushRadiosityHistory(primitives[idx].getRadiosity());
}


#endif // FORM_FACTORS_H
