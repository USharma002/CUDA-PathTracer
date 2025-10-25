#ifndef HITABLELISTH
#define HITABLELISTH

#include "triangle.h"
#include "quad.h"
#include "surface_interaction_record.h"
#include "bvh.h"

class Scene {
public:
    __host__ __device__ Scene() : primitives(nullptr), list_size(0), 
                                   bvh_nodes(nullptr), bvh_indices(nullptr), use_bvh(false) {}
    
    __host__ __device__ Scene(Primitive *l, int n) 
        : primitives(l), list_size(n), bvh_nodes(nullptr), bvh_indices(nullptr), use_bvh(false) {}
    
    __host__ __device__ Scene(Primitive *l, int n, BVHNode* nodes, int* indices) 
        : primitives(l), list_size(n), bvh_nodes(nodes), bvh_indices(indices), use_bvh(true) {}
    
    // Legacy constructors for backwards compatibility (automatically wrap Triangles in Primitives on host)
    __host__ Scene(Triangle *triangles, int n, bool legacy_mode) 
        : list_size(n), bvh_nodes(nullptr), bvh_indices(nullptr), use_bvh(false) {
        // This constructor should not be used on device
        // Caller must convert Triangle* to Primitive* before passing to device
        primitives = nullptr;  // Set externally
    }
    
    __device__ bool intersect(const Ray& r, float t_min, float t_max, SurfaceInteractionRecord &si) const {
        #if FORCE_LINEAR_SEARCH
        // Force linear search for debugging
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
        #else
        if (use_bvh && bvh_nodes != nullptr) {
            // Use BVH acceleration
            bool hit = traverseBVH(r, t_min, t_max, si, bvh_nodes, primitives, bvh_indices);
            if (hit) {
                // BVH traversal already sets si, but we need to set primitive info
                // Find which primitive was hit by checking the hit point
                for (int i = 0; i < list_size; i++) {
                    SurfaceInteractionRecord temp;
                    if (primitives[i].intersect(r, t_min, t_max, temp)) {
                        if (fabsf(temp.t - si.t) < 1e-6f) {
                            si.prim_ptr = &primitives[i];
                            si.hit_type = (primitives[i].type == PRIM_TRIANGLE) ? HIT_TRIANGLE : HIT_QUAD;
                            break;
                        }
                    }
                }
            }
            return hit;
        } else {
            // Fallback to linear search
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
        #endif
    }
    
    Primitive *primitives;  // Changed from Triangle* to Primitive*
    int list_size;
    
    // BVH data
    BVHNode* bvh_nodes;
    int* bvh_indices;
    bool use_bvh;
};

#endif
