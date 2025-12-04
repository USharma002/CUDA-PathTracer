#ifndef HITABLELISTH
#define HITABLELISTH

#include "rendering/triangle.h"
#include "rendering/quad.h"
#include "rendering/surface_interaction_record.h"
#include "rendering/bvh.h"

class Scene {
public:
    __host__ __device__ Scene() : primitives(nullptr), list_size(0), 
                                   filtered_formfactor(nullptr), filtered_radiosity(nullptr),
                                   precomputed_cdfs(nullptr),
                                   bvh_nodes(nullptr), bvh_indices(nullptr), use_bvh(false),
                                   use_filtered(false), mis_bsdf_fraction(0.5f) {}
    
    __host__ __device__ Scene(Primitive *l, int n) 
        : primitives(l), list_size(n), filtered_formfactor(nullptr), filtered_radiosity(nullptr),
          precomputed_cdfs(nullptr),
          bvh_nodes(nullptr), bvh_indices(nullptr), use_bvh(false), use_filtered(false),
          mis_bsdf_fraction(0.5f) {}
    
    __host__ __device__ Scene(Primitive *l, int n, BVHNode* nodes, int* indices) 
        : primitives(l), list_size(n), filtered_formfactor(nullptr), filtered_radiosity(nullptr),
          precomputed_cdfs(nullptr),
          bvh_nodes(nodes), bvh_indices(indices), use_bvh(true), use_filtered(false),
          mis_bsdf_fraction(0.5f) {}
    
    // Legacy constructors for backwards compatibility (automatically wrap Triangles in Primitives on host)
    __host__ Scene(Triangle *triangles, int n, bool legacy_mode) 
        : list_size(n), filtered_formfactor(nullptr), filtered_radiosity(nullptr),
          precomputed_cdfs(nullptr),
          bvh_nodes(nullptr), bvh_indices(nullptr), use_bvh(false), use_filtered(false) {
        // This constructor should not be used on device
        // Caller must convert Triangle* to Primitive* before passing to device
        primitives = nullptr;  // Set externally
    }
    
    __device__ bool intersect(const Ray& r, float t_min, float t_max, SurfaceInteractionRecord &si) const {
        if (use_bvh && bvh_nodes != nullptr) {
            // Use optimized BVH traversal
            return intersect_bvh_optimized(r, t_min, t_max, si);
        } else {
            // Fallback to linear search
            return intersect_linear(r, t_min, t_max, si);
        }
    }
    
    // Optimized BVH traversal with precomputed ray inverse direction
    __device__ bool intersect_bvh_optimized(const Ray& r, float t_min, float t_max, SurfaceInteractionRecord &si) const {
        bool hit_anything = false;
        float closest_t = t_max;
        
        // Stack for traversal
        int stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0;  // Root node
        
        // Precompute inverse direction for faster AABB tests
        Vector3f inv_dir(1.0f / r.direction().x(), 
                         1.0f / r.direction().y(), 
                         1.0f / r.direction().z());
        
        while (stack_ptr > 0) {
            int node_idx = stack[--stack_ptr];
            const BVHNode& node = bvh_nodes[node_idx];
            
            // Fast ray-box intersection
            float tmin_box = t_min;
            float tmax_box = closest_t;
            
            #pragma unroll
            for (int a = 0; a < 3; a++) {
                float t0 = (node.bbox.min[a] - r.origin()[a]) * inv_dir[a];
                float t1 = (node.bbox.max[a] - r.origin()[a]) * inv_dir[a];
                if (inv_dir[a] < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
                tmin_box = t0 > tmin_box ? t0 : tmin_box;
                tmax_box = t1 < tmax_box ? t1 : tmax_box;
            }
            
            if (tmax_box < tmin_box) continue;  // No intersection with box
            
            if (node.isLeaf()) {
                // Test all primitives in leaf
                for (int i = 0; i < node.prim_count; i++) {
                    int prim_idx = bvh_indices[node.left_child + i];
                    SurfaceInteractionRecord temp;
                    
                    if (primitives[prim_idx].intersect(r, t_min, closest_t, temp)) {
                        if (temp.t < closest_t) {
                            si = temp;
                            si.prim_ptr = &primitives[prim_idx];
                            si.hit_type = (primitives[prim_idx].type == PRIM_TRIANGLE) 
                                         ? HIT_TRIANGLE : HIT_QUAD;
                            closest_t = temp.t;
                            hit_anything = true;
                        }
                    }
                }
            } else {
                // Push children - ordered for better cache coherence
                if (stack_ptr < 62) {
                    stack[stack_ptr++] = node.right_child;
                    stack[stack_ptr++] = node.left_child;
                }
            }
        }
        
        return hit_anything;
    }
    
    // Linear search fallback
    __device__ bool intersect_linear(const Ray& r, float t_min, float t_max, SurfaceInteractionRecord &si) const {
        bool hit_anything = false;
        SurfaceInteractionRecord temp;
        si.t = t_max;
        
        for (int i = 0; i < list_size; i++) {
            if (primitives[i].intersect(r, t_min, si.t, temp)) {
                if (temp.t < si.t) {
                    si = temp;
                    si.prim_ptr = &primitives[i];
                    si.hit_type = (primitives[i].type == PRIM_TRIANGLE) ? HIT_TRIANGLE : HIT_QUAD;
                    hit_anything = true;
                }
            }
        }
        return hit_anything;
    }
    
    // Shadow ray - any hit (early termination)
    __device__ bool occluded(const Vector3f& from, const Vector3f& to) const {
        Vector3f dir = to - from;
        float dist = dir.length();
        dir = dir / dist;
        Ray shadow_ray(from + dir * 1e-4f, dir);
        
        if (use_bvh && bvh_nodes != nullptr) {
            return occluded_bvh(shadow_ray, 1e-4f, dist - 1e-4f);
        } else {
            SurfaceInteractionRecord temp;
            for (int i = 0; i < list_size; i++) {
                if (primitives[i].intersect(shadow_ray, 1e-4f, dist - 1e-4f, temp)) {
                    return true;
                }
            }
            return false;
        }
    }
    
    // Optimized shadow ray with BVH
    __device__ bool occluded_bvh(const Ray& r, float t_min, float t_max) const {
        int stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0;
        
        Vector3f inv_dir(1.0f / r.direction().x(), 
                         1.0f / r.direction().y(), 
                         1.0f / r.direction().z());
        
        while (stack_ptr > 0) {
            int node_idx = stack[--stack_ptr];
            const BVHNode& node = bvh_nodes[node_idx];
            
            float tmin_box = t_min, tmax_box = t_max;
            
            #pragma unroll
            for (int a = 0; a < 3; a++) {
                float t0 = (node.bbox.min[a] - r.origin()[a]) * inv_dir[a];
                float t1 = (node.bbox.max[a] - r.origin()[a]) * inv_dir[a];
                if (inv_dir[a] < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
                tmin_box = t0 > tmin_box ? t0 : tmin_box;
                tmax_box = t1 < tmax_box ? t1 : tmax_box;
            }
            
            if (tmax_box < tmin_box) continue;
            
            if (node.isLeaf()) {
                for (int i = 0; i < node.prim_count; i++) {
                    int prim_idx = bvh_indices[node.left_child + i];
                    SurfaceInteractionRecord temp;
                    if (primitives[prim_idx].intersect(r, t_min, t_max, temp)) {
                        return true;  // Early exit on any hit
                    }
                }
            } else {
                if (stack_ptr < 62) {
                    stack[stack_ptr++] = node.right_child;
                    stack[stack_ptr++] = node.left_child;
                }
            }
        }
        
        return false;
    }
    
    Primitive *primitives;  // Changed from Triangle* to Primitive*
    int list_size;
    // Optional device pointers to filtered PDFs (allocated/managed by host SceneState)
    float* filtered_formfactor; // contiguous block: num_prims * GRID_SIZE floats
    float* filtered_radiosity;  // contiguous block: num_prims * GRID_SIZE floats (luminance)
    
    // PRE-COMPUTED CDFs - Eliminates per-sample CDF building!
    // This is the key optimization: CDFs are built once on CPU when radiosity is updated
    PrecomputedCDF* precomputed_cdfs;  // device pointer: num_prims PrecomputedCDF structs
    
    // BVH data
    BVHNode* bvh_nodes;
    int* bvh_indices;
    bool use_bvh;
    
    // Flag to use filtered PDFs for sampling (toggled by UI checkbox)
    bool use_filtered;
    
    // MIS configuration: fraction of samples from BSDF (0.0 = all grid, 1.0 = all BSDF)
    float mis_bsdf_fraction;
};

#endif
