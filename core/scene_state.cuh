#ifndef SCENE_STATE_CUH
#define SCENE_STATE_CUH

#include <string>
#include <vector>
#include <chrono>
#include <iostream>

#include "../primitive.h"
#include "../bvh.h"
#include "../scene.h"
#include "../file_manager.h"
#include "cuda_utils.cuh"

// Forward declaration for OptiX support
struct OptixState;

// ============================================================================
// SCENE STATE MANAGEMENT
// ============================================================================

/**
 * @brief Manages scene geometry, materials, and acceleration structures
 */
struct SceneState {
    // Host-side primitives
    Primitive* h_primitives;
    
    // Device-side primitives
    Primitive* d_primitives;
    
    // BVH acceleration structure
    BVHNode* d_bvh_nodes;
    int* d_bvh_indices;
    
    // Scene object
    Scene* d_scene;
    
    // Scene info
    int num_primitives;
    std::string scene_file;
    
    // Statistics
    int num_triangles;
    int num_quads;
    
    // OptiX state (optional)
    OptixState* optix_state;
    bool use_optix;
    
    /**
     * @brief Constructor with default initialization
     */
    SceneState() 
        : h_primitives(nullptr)
        , d_primitives(nullptr)
        , d_bvh_nodes(nullptr)
        , d_bvh_indices(nullptr)
        , d_scene(nullptr)
        , num_primitives(0)
        , scene_file("./scenes/cbox_quads.obj")
        , num_triangles(0)
        , num_quads(0)
        , optix_state(nullptr)
        , use_optix(false)
    {}
    
    /**
     * @brief Load scene from OBJ file
     * @param filename Path to OBJ file
     * @param subdivision_count Number of subdivision iterations
     * @param convert_to_triangles Convert quads to triangles
     * @return true if loading succeeded
     */
    bool loadScene(const std::string& filename, int subdivision_count = 0, 
                   bool convert_to_triangles = false);
    
    /**
     * @brief Build acceleration structure
     * @param use_optix_accel Use OptiX acceleration structure
     * @return true if building succeeded
     */
    bool buildAccelStructure(bool use_optix_accel = false);
    
    /**
     * @brief Get scene statistics
     */
    void printStatistics() const;
    
    /**
     * @brief Cleanup all resources
     */
    void cleanup();
    
    /**
     * @brief Check if scene is loaded
     * @return true if scene is loaded
     */
    bool isLoaded() const { return num_primitives > 0; }
    
    /**
     * @brief Get memory usage in bytes
     * @return Memory usage
     */
    size_t getMemoryUsage() const;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Convert quads to triangles
 * @param primitives Input primitives
 * @return Triangulated primitives
 */
std::vector<Primitive> convertQuadsToTriangles(const std::vector<Primitive>& primitives);

#endif // SCENE_STATE_CUH
