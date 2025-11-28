#ifndef RADIOSITY_RENDERER_CUH
#define RADIOSITY_RENDERER_CUH

/**
 * @file radiosity_renderer.cuh
 * @brief Radiosity-based rendering implementation
 * 
 * This file contains radiosity rendering algorithms including:
 * - Direct radiosity visualization
 * - Delta radiosity (change visualization)
 * - History comparison
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../vector.h"
#include "../ray.h"
#include "../scene.h"
#include "../sensor.h"
#include "../surface_interaction_record.h"

// ============================================================================
// RADIOSITY RENDERER CONFIGURATION
// ============================================================================

namespace radiosity_renderer {

// Configuration constants
constexpr int DEFAULT_MAX_DEPTH = 1;  // Radiosity is usually direct illumination only
constexpr float DELTA_SCALE = 5.0f;   // Scale factor for delta visualization

} // namespace radiosity_renderer

// ============================================================================
// RADIOSITY INTEGRATOR
// ============================================================================

/**
 * @brief Radiosity integrator kernel
 * 
 * Simple integrator that visualizes precomputed radiosity values.
 * 
 * @param s Scene pointer
 * @param ray_ Initial ray
 * @param L Output radiance (accumulated)
 * @param max_depth Maximum depth (usually 1 for radiosity)
 * @param rand_state Random state
 */
__device__ void radiosityIntegrator(
    const Scene* s, 
    Ray& ray_, 
    Vector3f& L, 
    int max_depth,
    curandState* rand_state);

// ============================================================================
// RADIOSITY RENDER KERNELS
// ============================================================================

/**
 * @brief Main render kernel for radiosity visualization
 * 
 * @param image Output image buffer
 * @param cam Camera sensor
 * @param scene Scene data
 * @param rand_state Random states per pixel
 * @param spp Samples per pixel
 */
__global__ void radiosityRender(
    unsigned char* image, 
    Sensor* cam, 
    Scene* scene,
    curandState* rand_state, 
    int spp);

/**
 * @brief Delta radiosity visualization kernel
 * 
 * Visualizes the difference between two radiosity history steps.
 * Useful for understanding light transport convergence.
 * 
 * @param image Output image buffer
 * @param width Image width
 * @param height Image height
 * @param camera Camera sensor
 * @param scene Scene data
 * @param step1 First history step (more recent)
 * @param step2 Second history step (older)
 */
__global__ void radiosityDeltaRender(
    unsigned char* image, 
    int width, 
    int height,
    Sensor* camera, 
    Scene* scene, 
    int step1, 
    int step2);

/**
 * @brief Radiosity history visualization kernel
 * 
 * Visualizes a specific step in the radiosity history.
 * 
 * @param image Output image buffer
 * @param width Image width
 * @param height Image height
 * @param camera Camera sensor
 * @param scene Scene data
 * @param history_step History step to visualize (0 = most recent)
 */
__global__ void radiosityHistoryRender(
    unsigned char* image,
    int width,
    int height,
    Sensor* camera,
    Scene* scene,
    int history_step);

// ============================================================================
// RADIOSITY SOLVER KERNELS
// ============================================================================

/**
 * @brief Initialize directional grids for all primitives
 * 
 * @param primitives Primitive array
 * @param num_primitives Number of primitives
 */
__global__ void initializeDirectionalGrids(
    Primitive* primitives, 
    int num_primitives);

/**
 * @brief Initialize random states for form factor computation
 * 
 * @param num_pairs Number of primitive pairs
 * @param rand_states Output random states
 */
__global__ void formFactorRandInit(
    int num_pairs, 
    curandState* rand_states);

/**
 * @brief Calculate form factors using Monte Carlo integration
 * 
 * @param form_factors Output form factor matrix
 * @param primitives Primitive array
 * @param num_primitives Number of primitives
 * @param n_samples Number of Monte Carlo samples
 * @param rand_states Random states
 * @param bvh_nodes BVH nodes for visibility
 * @param bvh_indices BVH primitive indices
 */
__global__ void calculateFormFactorsMC(
    float* form_factors, 
    Primitive* primitives,
    int num_primitives, 
    int n_samples,
    curandState* rand_states, 
    BVHNode* bvh_nodes,
    int* bvh_indices);

/**
 * @brief Calculate form factors using point-to-point method
 * 
 * @param form_factors Output form factor matrix
 * @param primitives Primitive array
 * @param num_primitives Number of primitives
 * @param bvh_nodes BVH nodes for visibility
 * @param bvh_indices BVH primitive indices
 */
__global__ void calculateFormFactorsPointToPoint(
    float* form_factors, 
    Primitive* primitives, 
    int num_primitives, 
    BVHNode* bvh_nodes,
    int* bvh_indices);

/**
 * @brief Perform one radiosity iteration (gathering)
 * 
 * @param primitives Primitive array
 * @param form_factors Form factor matrix
 * @param num_primitives Number of primitives
 */
__global__ void radiosityIteration(
    Primitive* primitives, 
    float* form_factors, 
    int num_primitives);

/**
 * @brief Update radiosity grids from form factors
 * 
 * @param primitives Primitive array
 * @param form_factors Form factor matrix
 * @param num_primitives Number of primitives
 */
__global__ void updateRadiosityGrid(
    Primitive* primitives, 
    float* form_factors, 
    int num_primitives);

/**
 * @brief Store current radiosity to history buffer
 * 
 * @param primitives Primitive array
 * @param num_prims Number of primitives
 */
__global__ void storeRadiosityHistory(
    Primitive* primitives, 
    int num_prims);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Convert direction to grid indices
 * 
 * @param dir Direction vector
 * @param grid_theta Output theta index
 * @param grid_phi Output phi index
 */
__device__ void directionToGridIndices(
    const Vector3f& dir, 
    int& grid_theta, 
    int& grid_phi);

/**
 * @brief Test visibility between two points
 * 
 * @param ray Shadow ray
 * @param max_dist Maximum distance
 * @param bvh_nodes BVH nodes
 * @param bvh_indices BVH indices
 * @param primitives Primitive array
 * @param num_primitives Number of primitives
 * @param target_idx Target primitive index
 * @param source_idx Source primitive index
 * @return true if visible
 */
__device__ bool visibilityTest(
    const Ray& ray, 
    float max_dist,
    const BVHNode* bvh_nodes, 
    const int* bvh_indices,
    const Primitive* primitives, 
    int num_primitives,
    int target_idx, 
    int source_idx);

#endif // RADIOSITY_RENDERER_CUH
