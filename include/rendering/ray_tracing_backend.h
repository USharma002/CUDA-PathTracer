/**
 * @file ray_tracing_backend.h
 * @brief Unified ray tracing interface supporting both OptiX and software BVH
 * 
 * This provides a clean abstraction over:
 * - OptiX 7+ hardware-accelerated ray tracing (RTX GPUs)
 * - Software BVH fallback for any CUDA-capable GPU
 * 
 * The integrator uses this interface without knowing which backend is active.
 */

#ifndef RAY_TRACING_BACKEND_H
#define RAY_TRACING_BACKEND_H

#include "rendering/bvh.h"
#include "rendering/primitive.h"
#include "rendering/scene.h"
#include "rendering/surface_interaction_record.h"
#include "utils/optix_logger.h"

// Check for OptiX at compile time
#ifdef USE_OPTIX
#include "rendering/optix_wrapper.h"
#endif

// ============================================================================
// RAY TRACING BACKEND SELECTION
// ============================================================================

enum class RayTracingBackend {
    SOFTWARE_BVH,    // Our custom CUDA BVH
    OPTIX_RTX        // NVIDIA OptiX (requires RTX GPU)
};

// ============================================================================
// BACKEND MANAGER - Host-side singleton
// ============================================================================

class RayTracingManager {
public:
    static RayTracingManager& getInstance() {
        static RayTracingManager instance;
        return instance;
    }
    
    /**
     * Initialize the ray tracing backend
     * Tries OptiX first, falls back to software BVH
     */
    bool initialize(bool prefer_optix = true) {
        auto& log = OptixLogger::getInstance();
        m_backend = RayTracingBackend::SOFTWARE_BVH;
        
        log.info("RTBackend", "========== Ray Tracing Backend Initialization ==========");
        
#ifdef USE_OPTIX
        if (prefer_optix) {
            log.info("RTBackend", "OptiX support enabled, attempting initialization...");
            if (m_optix.initialize()) {
                m_backend = RayTracingBackend::OPTIX_RTX;
                log.info("RTBackend", "Using OptiX RTX hardware acceleration");
                log.info("RTBackend", "=========================================================");
                return true;
            }
            log.warning("RTBackend", "OptiX initialization failed, falling back to software BVH");
        } else {
            log.info("RTBackend", "OptiX disabled by preference");
        }
#else
        log.info("RTBackend", "OptiX support not compiled (USE_OPTIX not defined)");
#endif
        
        log.info("RTBackend", "Using software BVH for ray tracing");
        log.info("RTBackend", "=========================================================");
        return true;
    }
    
    /**
     * Build acceleration structure
     */
    bool buildAccelStructure(Primitive* primitives, int num_primitives,
                             BVHNode** out_bvh_nodes, int** out_bvh_indices,
                             int& out_num_nodes) {
        auto& log = OptixLogger::getInstance();
        
        log.info("RTBackend", "Building acceleration structures...");
        
        // Always build software BVH (needed for fallback and radiosity)
        log.debug("RTBackend", "Building software BVH...");
        auto bvh_start = std::chrono::high_resolution_clock::now();
        
        BVHBuilder builder(primitives, num_primitives);
        
        // Copy to device
        size_t nodes_size = builder.nodes.size() * sizeof(BVHNode);
        size_t indices_size = builder.primitive_indices.size() * sizeof(int);
        
        cudaMalloc(out_bvh_nodes, nodes_size);
        cudaMalloc(out_bvh_indices, indices_size);
        cudaMemcpy(*out_bvh_nodes, builder.nodes.data(), nodes_size, cudaMemcpyHostToDevice);
        cudaMemcpy(*out_bvh_indices, builder.primitive_indices.data(), indices_size, cudaMemcpyHostToDevice);
        out_num_nodes = builder.nodes.size();
        
        auto bvh_end = std::chrono::high_resolution_clock::now();
        double bvh_ms = std::chrono::duration<double, std::milli>(bvh_end - bvh_start).count();
        
        std::stringstream ss;
        ss << "Software BVH built: " << builder.nodes.size() << " nodes, " 
           << num_primitives << " primitives in " << std::fixed << std::setprecision(2) 
           << bvh_ms << " ms";
        log.info("RTBackend", ss.str());
        
        log.logMemoryAlloc("bvh_nodes", nodes_size);
        log.logMemoryAlloc("bvh_indices", indices_size);
        
#ifdef USE_OPTIX
        if (m_backend == RayTracingBackend::OPTIX_RTX) {
            log.info("RTBackend", "Building OptiX acceleration structure...");
            if (!m_optix.buildAccelStructure(primitives, num_primitives)) {
                log.warning("RTBackend", "OptiX build failed, using software BVH only");
                m_backend = RayTracingBackend::SOFTWARE_BVH;
            } else {
                log.info("RTBackend", "OptiX acceleration structure ready");
            }
        }
#endif
        
        return true;
    }
    
    RayTracingBackend getBackend() const { return m_backend; }
    bool isOptixAvailable() const { 
#ifdef USE_OPTIX
        return m_optix.isAvailable();
#else
        return false;
#endif
    }
    
    const char* getBackendName() const {
        switch (m_backend) {
            case RayTracingBackend::OPTIX_RTX: return "OptiX RTX";
            case RayTracingBackend::SOFTWARE_BVH: return "Software BVH";
            default: return "Unknown";
        }
    }
    
#ifdef USE_OPTIX
    OptixWrapper& getOptix() { return m_optix; }
#endif
    
    void cleanup() {
        auto& log = OptixLogger::getInstance();
        log.info("RTBackend", "Cleaning up ray tracing backend...");
        
#ifdef USE_OPTIX
        if (m_backend == RayTracingBackend::OPTIX_RTX) {
            log.info("RTBackend", "Cleaning up OptiX resources...");
            m_optix.cleanup();
        }
#endif
        
        log.info("RTBackend", "Ray tracing backend cleanup complete");
    }
    
    void logStats() {
#ifdef USE_OPTIX
        if (m_backend == RayTracingBackend::OPTIX_RTX) {
            m_optix.logStats();
        }
#endif
    }
    
private:
    RayTracingManager() : m_backend(RayTracingBackend::SOFTWARE_BVH) {}
    ~RayTracingManager() { cleanup(); }
    
    RayTracingBackend m_backend;
    
#ifdef USE_OPTIX
    OptixWrapper m_optix;
#endif
};

// ============================================================================
// OPTIMIZED SOFTWARE BVH TRAVERSAL
// ============================================================================

/**
 * Optimized BVH traversal with better memory access patterns
 * This is an improved version of the existing traverseBVH function
 */
__device__ bool traverseBVH_optimized(
    const Ray& r, 
    float t_min, 
    float t_max, 
    SurfaceInteractionRecord& si,
    const BVHNode* __restrict__ nodes, 
    const Primitive* __restrict__ primitives,
    const int* __restrict__ primitive_indices) 
{
    bool hit_anything = false;
    float closest_t = t_max;
    
    // Use shared memory for stack if block size allows
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // Start with root
    
    // Precompute ray inverse direction for faster AABB tests
    Vector3f inv_dir(1.0f / r.direction().x(), 
                     1.0f / r.direction().y(), 
                     1.0f / r.direction().z());
    int dir_is_neg[3] = { inv_dir.x() < 0, inv_dir.y() < 0, inv_dir.z() < 0 };
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = nodes[node_idx];
        
        // Fast ray-box intersection using precomputed inverse direction
        float tmin_box = t_min;
        float tmax_box = closest_t;
        
        for (int a = 0; a < 3; a++) {
            float t0 = (node.bbox.min[a] - r.origin()[a]) * inv_dir[a];
            float t1 = (node.bbox.max[a] - r.origin()[a]) * inv_dir[a];
            if (dir_is_neg[a]) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin_box = t0 > tmin_box ? t0 : tmin_box;
            tmax_box = t1 < tmax_box ? t1 : tmax_box;
            if (tmax_box < tmin_box) goto skip_node;
        }
        
        if (node.isLeaf()) {
            // Test all primitives in leaf
            for (int i = 0; i < node.prim_count; i++) {
                int prim_idx = primitive_indices[node.left_child + i];
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
            // Push children in front-to-back order based on ray direction
            // This improves early termination
            if (stack_ptr < 62) {
                // Simple heuristic: push farther child first so closer child is processed first
                const BVHNode& left = nodes[node.left_child];
                const BVHNode& right = nodes[node.right_child];
                
                float left_dist = (left.bbox.center() - r.origin()).length();
                float right_dist = (right.bbox.center() - r.origin()).length();
                
                if (left_dist < right_dist) {
                    stack[stack_ptr++] = node.right_child;
                    stack[stack_ptr++] = node.left_child;
                } else {
                    stack[stack_ptr++] = node.left_child;
                    stack[stack_ptr++] = node.right_child;
                }
            }
        }
        
        skip_node:;
    }
    
    return hit_anything;
}

/**
 * Any-hit query (shadow rays) - early termination on first hit
 */
__device__ bool traverseBVH_anyhit(
    const Ray& r, 
    float t_min, 
    float t_max,
    const BVHNode* __restrict__ nodes, 
    const Primitive* __restrict__ primitives,
    const int* __restrict__ primitive_indices) 
{
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    Vector3f inv_dir(1.0f / r.direction().x(), 
                     1.0f / r.direction().y(), 
                     1.0f / r.direction().z());
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = nodes[node_idx];
        
        // Fast ray-box intersection
        float tmin_box = t_min;
        float tmax_box = t_max;
        
        for (int a = 0; a < 3; a++) {
            float t0 = (node.bbox.min[a] - r.origin()[a]) * inv_dir[a];
            float t1 = (node.bbox.max[a] - r.origin()[a]) * inv_dir[a];
            if (inv_dir[a] < 0) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin_box = t0 > tmin_box ? t0 : tmin_box;
            tmax_box = t1 < tmax_box ? t1 : tmax_box;
            if (tmax_box < tmin_box) goto skip;
        }
        
        if (node.isLeaf()) {
            for (int i = 0; i < node.prim_count; i++) {
                int prim_idx = primitive_indices[node.left_child + i];
                SurfaceInteractionRecord temp;
                if (primitives[prim_idx].intersect(r, t_min, t_max, temp)) {
                    return true;  // Early exit!
                }
            }
        } else {
            if (stack_ptr < 62) {
                stack[stack_ptr++] = node.right_child;
                stack[stack_ptr++] = node.left_child;
            }
        }
        
        skip:;
    }
    
    return false;
}

// ============================================================================
// SCENE INTERSECTION USING SELECTED BACKEND
// ============================================================================

/**
 * Updated Scene class intersection that can use optimized BVH
 * This replaces the existing intersect method
 */
__device__ bool scene_intersect_optimized(
    const Scene* scene,
    const Ray& r, 
    float t_min, 
    float t_max, 
    SurfaceInteractionRecord& si)
{
    if (scene->use_bvh && scene->bvh_nodes != nullptr) {
        return traverseBVH_optimized(r, t_min, t_max, si, 
                                     scene->bvh_nodes, 
                                     scene->primitives,
                                     scene->bvh_indices);
    } else {
        // Fallback to linear search
        bool hit_anything = false;
        SurfaceInteractionRecord temp;
        si.t = t_max;
        
        for (int i = 0; i < scene->list_size; i++) {
            if (scene->primitives[i].intersect(r, t_min, si.t, temp)) {
                if (temp.t < si.t) {
                    si = temp;
                    si.prim_ptr = &scene->primitives[i];
                    si.hit_type = (scene->primitives[i].type == PRIM_TRIANGLE) 
                                 ? HIT_TRIANGLE : HIT_QUAD;
                    hit_anything = true;
                }
            }
        }
        return hit_anything;
    }
}

/**
 * Shadow ray query
 */
__device__ bool scene_occluded(
    const Scene* scene,
    const Vector3f& from,
    const Vector3f& to)
{
    Vector3f dir = to - from;
    float dist = dir.length();
    dir = dir / dist;
    
    Ray shadow_ray(from + dir * 1e-4f, dir);
    
    if (scene->use_bvh && scene->bvh_nodes != nullptr) {
        return traverseBVH_anyhit(shadow_ray, 1e-4f, dist - 1e-4f,
                                  scene->bvh_nodes,
                                  scene->primitives,
                                  scene->bvh_indices);
    } else {
        SurfaceInteractionRecord temp;
        for (int i = 0; i < scene->list_size; i++) {
            if (scene->primitives[i].intersect(shadow_ray, 1e-4f, dist - 1e-4f, temp)) {
                return true;
            }
        }
        return false;
    }
}

#endif // RAY_TRACING_BACKEND_H
