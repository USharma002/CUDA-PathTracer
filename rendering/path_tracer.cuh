#ifndef PATH_TRACER_CUH
#define PATH_TRACER_CUH

/**
 * @file path_tracer.cuh
 * @brief Path tracing integrator implementation
 * 
 * This file contains the path tracing algorithm implementation including:
 * - Primary ray generation
 * - Path integration with multiple bounces
 * - Multiple importance sampling
 * - Russian roulette termination
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../vector.h"
#include "../ray.h"
#include "../scene.h"
#include "../sensor.h"
#include "../surface_interaction_record.h"
#include "../integrator.h"

// ============================================================================
// PATH TRACER CONFIGURATION
// ============================================================================

namespace path_tracer {

// Default configuration
constexpr int DEFAULT_MAX_DEPTH = 10;
constexpr float MIN_THROUGHPUT = 1e-4f;
constexpr float RR_START_DEPTH = 3;
constexpr float RR_MIN_PROB = 0.05f;

} // namespace path_tracer

// ============================================================================
// PATH TRACING INTEGRATOR
// ============================================================================

/**
 * @brief Path tracing integrator kernel
 * 
 * Implements unbiased Monte Carlo path tracing with:
 * - Cosine-weighted hemisphere sampling (BSDF)
 * - Grid-based importance sampling
 * - Multiple importance sampling (MIS)
 * - Russian roulette termination
 * 
 * @param s Scene pointer
 * @param ray_ Initial ray
 * @param L Output radiance (accumulated)
 * @param max_depth Maximum path depth
 * @param rand_state Random state for sampling
 * @param sampling_mode Sampling strategy to use
 */
__device__ void pathTracingIntegrator(
    const Scene* s, 
    Ray& ray_, 
    Vector3f& L, 
    int max_depth,
    curandState* rand_state, 
    SamplingMode sampling_mode);

/**
 * @brief Main render kernel for path tracing
 * 
 * @param image Output image buffer (RGB, unsigned char)
 * @param cam Camera sensor
 * @param scene Scene data
 * @param rand_state Random states per pixel
 * @param spp Samples per pixel
 * @param sampling_mode Sampling strategy
 */
__global__ void pathTracingRender(
    unsigned char* image, 
    Sensor* cam, 
    Scene* scene,
    curandState* rand_state, 
    int spp, 
    SamplingMode sampling_mode);

/**
 * @brief Progressive rendering kernel
 * 
 * Accumulates samples over multiple frames for progressive refinement.
 * 
 * @param image Output image buffer
 * @param accum Accumulation buffer (float3)
 * @param cam Camera sensor
 * @param scene Scene data
 * @param rand_state Random states
 * @param frame_number Current frame number
 * @param sampling_mode Sampling strategy
 */
__global__ void pathTracingProgressiveRender(
    unsigned char* image,
    float* accum,
    Sensor* cam,
    Scene* scene,
    curandState* rand_state,
    int frame_number,
    SamplingMode sampling_mode);

// ============================================================================
// SAMPLING UTILITIES
// ============================================================================

/**
 * @brief Sample cosine-weighted hemisphere direction
 * 
 * Uses Malley's method for efficient cosine-weighted sampling.
 * 
 * @param normal Surface normal
 * @param rand_state Random state
 * @return Sampled direction in world space
 */
__device__ Vector3f sampleCosineWeightedHemisphere(
    const Vector3f& normal, 
    curandState* rand_state);

/**
 * @brief Sample direction from grid-based importance function
 * 
 * @param grid Visibility/radiosity grid
 * @param normal Surface normal
 * @param rand_state Random state
 * @param success Output flag indicating success
 * @return Sampled direction in world space
 */
__device__ Vector3f sampleGridDirection(
    const float* grid, 
    const Vector3f& normal,
    curandState* rand_state, 
    bool& success);

/**
 * @brief Sample with multiple importance sampling
 * 
 * Combines BSDF and grid sampling with balance heuristic.
 * 
 * @param grid Visibility/radiosity grid
 * @param normal Surface normal
 * @param rand_state Random state
 * @param weight Output MIS weight
 * @return Sampled direction in world space
 */
__device__ Vector3f sampleMIS(
    const float* grid, 
    const Vector3f& normal,
    curandState* rand_state, 
    float& weight);

// ============================================================================
// TONE MAPPING
// ============================================================================

/**
 * @brief Apply ACES filmic tone mapping
 * 
 * @param color Linear HDR color
 * @return Tone mapped color
 */
__device__ __host__ inline Vector3f toneMapACES(const Vector3f& color) {
    // ACES approximation
    Vector3f mapped = color / (color + Vector3f(1.0f, 1.0f, 1.0f));
    return mapped;
}

/**
 * @brief Apply gamma correction
 * 
 * @param color Linear color
 * @param gamma Gamma value (default 2.2)
 * @return Gamma corrected color
 */
__device__ __host__ inline Vector3f gammaCorrect(const Vector3f& color, float gamma = 2.2f) {
    float inv_gamma = 1.0f / gamma;
    return Vector3f(
        powf(color.x(), inv_gamma),
        powf(color.y(), inv_gamma),
        powf(color.z(), inv_gamma)
    );
}

/**
 * @brief Convert float color to unsigned char
 * 
 * @param color Float color (0-1 range)
 * @return Unsigned char color (0-255)
 */
__device__ __host__ inline void colorToUchar(const Vector3f& color, unsigned char* out) {
    out[0] = static_cast<unsigned char>(255.99f * fminf(color.x(), 1.0f));
    out[1] = static_cast<unsigned char>(255.99f * fminf(color.y(), 1.0f));
    out[2] = static_cast<unsigned char>(255.99f * fminf(color.z(), 1.0f));
}

#endif // PATH_TRACER_CUH
