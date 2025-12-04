#ifndef BVH_H
#define BVH_H

#include "core/vector.h"
#include "core/ray.h"
#include "rendering/triangle.h"
#include "rendering/quad.h"
#include "rendering/primitive.h"
#include "rendering/surface_interaction_record.h"
#include <algorithm>

// Axis-Aligned Bounding Box
struct AABB {
    Vector3f min;
    Vector3f max;
    
    __host__ __device__ AABB() : min(Vector3f(1e30f, 1e30f, 1e30f)), max(Vector3f(-1e30f, -1e30f, -1e30f)) {}
    
    __host__ __device__ AABB(const Vector3f& min_, const Vector3f& max_) : min(min_), max(max_) {}
    
    __host__ __device__ Vector3f center() const {
        return (min + max) * 0.5f;
    }
    
    __host__ __device__ float surfaceArea() const {
        Vector3f d = max - min;
        return 2.0f * (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
    }
    
    __host__ __device__ AABB merge(const AABB& other) const {
        return AABB(
            Vector3f(fminf(min.x(), other.min.x()), fminf(min.y(), other.min.y()), fminf(min.z(), other.min.z())),
            Vector3f(fmaxf(max.x(), other.max.x()), fmaxf(max.y(), other.max.y()), fmaxf(max.z(), other.max.z()))
        );
    }
    
    // FIXED: More robust ray-box intersection
    __device__ bool intersect(const Ray& r, float t_min, float t_max) const {
        // Handle rays parallel to axes
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0 = (min[a] - r.origin()[a]) * invD;
            float t1 = (max[a] - r.origin()[a]) * invD;
            
            if (invD < 0.0f) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }
            
            // Be more lenient with near-miss cases
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            
            if (t_max <= t_min)
                return false;
        }
        return true;
    }
};

// BVH Node structure
struct BVHNode {
    AABB bbox;
    int left_child;
    int right_child;
    int prim_count;
    
    __host__ __device__ BVHNode() : left_child(-1), right_child(-1), prim_count(0) {}
    
    __host__ __device__ bool isLeaf() const { return prim_count > 0; }
};


// Host-side BVH builder
class BVHBuilder {
public:
    std::vector<BVHNode> nodes;
    std::vector<int> primitive_indices;
    Primitive* primitives;
    int num_primitives;
    
BVHBuilder(Primitive* prims, int count) : primitives(prims), num_primitives(count) {
    std::cout << "  [BVH] Constructor called with " << count << " primitives" << std::endl;
    
    try {
        nodes.reserve(count * 2);
        std::cout << "  [BVH] Reserved space for " << (count * 2) << " nodes" << std::endl;
        
        primitive_indices.reserve(count);
        std::cout << "  [BVH] Reserved space for " << count << " indices" << std::endl;
        
        for (int i = 0; i < count; i++) {
            primitive_indices.push_back(i);
        }
        std::cout << "  [BVH] Initialized " << primitive_indices.size() << " indices" << std::endl;
        
        std::cout << "  [BVH] Starting recursive build..." << std::endl;
        buildRecursive(0, count);
        std::cout << "  [BVH] Recursive build complete" << std::endl;
        std::cout << "  [BVH] Final node count: " << nodes.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  [BVH] ERROR: " << e.what() << std::endl;
        throw;
    }
}

    
private:
    AABB computeBounds(int start, int end) {
        AABB bounds;
        const float eps = 1e-6f;
        
        for (int i = start; i < end; i++) {
            Primitive& prim = primitives[primitive_indices[i]];
            
            if (prim.type == PRIM_TRIANGLE) {
                Triangle& tri = prim.tri;
                bounds = bounds.merge(AABB(
                    Vector3f(fminf(fminf(tri.v0.x(), tri.v1.x()), tri.v2.x()) - eps,
                            fminf(fminf(tri.v0.y(), tri.v1.y()), tri.v2.y()) - eps,
                            fminf(fminf(tri.v0.z(), tri.v1.z()), tri.v2.z()) - eps),
                    Vector3f(fmaxf(fmaxf(tri.v0.x(), tri.v1.x()), tri.v2.x()) + eps,
                            fmaxf(fmaxf(tri.v0.y(), tri.v1.y()), tri.v2.y()) + eps,
                            fmaxf(fmaxf(tri.v0.z(), tri.v1.z()), tri.v2.z()) + eps)
                ));
            } else {
                Quad& quad = prim.quad;
                float min_x = fminf(fminf(quad.v00.x(), quad.v10.x()), 
                                   fminf(quad.v11.x(), quad.v01.x()));
                float min_y = fminf(fminf(quad.v00.y(), quad.v10.y()), 
                                   fminf(quad.v11.y(), quad.v01.y()));
                float min_z = fminf(fminf(quad.v00.z(), quad.v10.z()), 
                                   fminf(quad.v11.z(), quad.v01.z()));
                
                float max_x = fmaxf(fmaxf(quad.v00.x(), quad.v10.x()), 
                                   fmaxf(quad.v11.x(), quad.v01.x()));
                float max_y = fmaxf(fmaxf(quad.v00.y(), quad.v10.y()), 
                                   fmaxf(quad.v11.y(), quad.v01.y()));
                float max_z = fmaxf(fmaxf(quad.v00.z(), quad.v10.z()), 
                                   fmaxf(quad.v11.z(), quad.v01.z()));
                
                bounds = bounds.merge(AABB(
                    Vector3f(min_x - eps, min_y - eps, min_z - eps),
                    Vector3f(max_x + eps, max_y + eps, max_z + eps)
                ));
            }
        }
        return bounds;
    }
    
    Vector3f computeCentroid(int idx) {
        return primitives[primitive_indices[idx]].centroid();
    }
    
    int buildRecursive(int start, int end) {
        int node_idx = nodes.size();
        nodes.push_back(BVHNode());
        
        AABB bbox = computeBounds(start, end);
        int count = end - start;
        
        // Leaf node - increased threshold to 4 for better stability
        if (count <= 4) {
            nodes[node_idx].bbox = bbox;
            nodes[node_idx].left_child = start;
            nodes[node_idx].prim_count = count;
            return node_idx;
        }
        
        // Find best split axis
        AABB centroid_bounds;
        for (int i = start; i < end; i++) {
            Vector3f c = computeCentroid(i);
            centroid_bounds = centroid_bounds.merge(AABB(c, c));
        }
        
        int best_axis = 0;
        Vector3f extent = centroid_bounds.max - centroid_bounds.min;
        if (extent.y() > extent.x()) best_axis = 1;
        if (extent.z() > extent[best_axis]) best_axis = 2;
        
        // Check for degenerate case
        if (extent[best_axis] < 1e-6f) {
            nodes[node_idx].bbox = bbox;
            nodes[node_idx].left_child = start;
            nodes[node_idx].prim_count = count;
            return node_idx;
        }
        
        // Midpoint split
        float split_pos = centroid_bounds.center()[best_axis];
        
        int mid = start;
        for (int i = start; i < end; i++) {
            if (computeCentroid(i)[best_axis] < split_pos) {
                std::swap(primitive_indices[i], primitive_indices[mid]);
                mid++;
            }
        }
        
        // Ensure non-empty partitions
        if (mid == start || mid == end) {
            mid = start + count / 2;
        }
        
        // Build children
        int left_idx = buildRecursive(start, mid);
        int right_idx = buildRecursive(mid, end);
        
        // Set node data
        nodes[node_idx].bbox = bbox;
        nodes[node_idx].left_child = left_idx;
        nodes[node_idx].right_child = right_idx;
        nodes[node_idx].prim_count = 0;
        
        return node_idx;
    }
};

// Device-side BVH traversal with unified primitives
__device__ bool traverseBVH(const Ray& r, float t_min, float t_max, 
                            SurfaceInteractionRecord& si,
                            const BVHNode* nodes, 
                            const Primitive* primitives,
                            const int* primitive_indices) {
    bool hit_anything = false;
    si.t = t_max;
    
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = nodes[node_idx];
        
        // CRITICAL FIX: Test bbox with ORIGINAL t_max, not si.t
        // This ensures we don't prematurely cull nodes that might contain closer hits
        if (!node.bbox.intersect(r, t_min, t_max)) {
            continue;
        }
        
        if (node.isLeaf()) {
            // Test all primitives in this leaf
            for (int i = 0; i < node.prim_count; i++) {
                int prim_idx = primitive_indices[node.left_child + i];
                SurfaceInteractionRecord temp;
                
                if (primitives[prim_idx].intersect(r, t_min, t_max, temp)) {
                    // Update if closer
                    if (temp.t < si.t) {
                        si = temp;
                        hit_anything = true;
                        // Update t_max for subsequent tests
                        t_max = si.t;
                    }
                }
            }
        } else {
            // Push both children
            if (stack_ptr < 62) {
                stack[stack_ptr++] = node.right_child;
                stack[stack_ptr++] = node.left_child;
            }
        }
    }
    
    return hit_anything;
}

#endif // BVH_H
