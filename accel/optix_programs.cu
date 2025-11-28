/**
 * @file optix_programs.cu
 * @brief OptiX ray tracing programs
 * 
 * This file contains the OptiX device programs for hardware-accelerated ray tracing:
 * - Ray generation program
 * - Miss program  
 * - Closest hit program
 * - Any hit program (for shadows)
 * 
 * These programs are compiled to PTX and loaded at runtime when OptiX is enabled.
 */

#include "optix_types.cuh"
#include "../vector.h"
#include "../primitive.h"

#if ENABLE_OPTIX

#include <optix_device.h>

// ============================================================================
// CONSTANTS
// ============================================================================

// Numerical stability threshold for coordinate system building
constexpr float NORMAL_EPSILON = 0.999f;

// Gamma correction value for sRGB output
constexpr float GAMMA_CORRECTION = 2.2f;

// Ray tracing constants
constexpr float RAY_TMIN = 0.001f;
constexpr float RAY_TMAX = 1e16f;

// Pi constant
constexpr float PI = 3.14159265359f;

// ============================================================================
// LAUNCH PARAMETERS
// ============================================================================

extern "C" {
__constant__ OptixLaunchParams params;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Get current ray payload
 */
static __forceinline__ __device__ void* getPayload() {
    return reinterpret_cast<void*>(
        static_cast<uint64_t>(optixGetPayload_0()) |
        (static_cast<uint64_t>(optixGetPayload_1()) << 32)
    );
}

/**
 * @brief Set ray payload
 */
static __forceinline__ __device__ void setPayload(void* ptr) {
    uint64_t val = reinterpret_cast<uint64_t>(ptr);
    optixSetPayload_0(static_cast<uint32_t>(val));
    optixSetPayload_1(static_cast<uint32_t>(val >> 32));
}

/**
 * @brief Pack two floats into a uint32
 */
static __forceinline__ __device__ uint32_t packFloat2(float a, float b) {
    uint16_t a16 = __float2half_rn(a);
    uint16_t b16 = __float2half_rn(b);
    return (static_cast<uint32_t>(b16) << 16) | a16;
}

// ============================================================================
// RAY PAYLOAD STRUCTURE
// ============================================================================

struct RayPayload {
    Vector3f color;
    Vector3f attenuation;
    Vector3f origin;
    Vector3f direction;
    int depth;
    bool done;
    unsigned int seed;
};

// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/**
 * @brief Simple LCG random number generator
 */
static __forceinline__ __device__ unsigned int lcg(unsigned int& prev) {
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = LCG_A * prev + LCG_C;
    return prev;
}

/**
 * @brief Generate random float in [0, 1)
 */
static __forceinline__ __device__ float randomFloat(unsigned int& seed) {
    return static_cast<float>(lcg(seed)) / static_cast<float>(0xFFFFFFFFu);
}

// ============================================================================
// RAY GENERATION PROGRAM
// ============================================================================

extern "C" __global__ void __raygen__rg() {
    // Get pixel coordinates
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    const int x = idx.x;
    const int y = idx.y;
    
    // Initialize random seed
    unsigned int seed = params.random_seed + y * params.width + x;
    
    // Accumulate color
    Vector3f pixel_color(0.0f, 0.0f, 0.0f);
    
    for (int s = 0; s < params.samples_per_pixel; s++) {
        // Generate ray with jitter
        const float u = (static_cast<float>(x) + randomFloat(seed)) / static_cast<float>(params.width);
        const float v = (static_cast<float>(y) + randomFloat(seed)) / static_cast<float>(params.height);
        
        Vector3f ray_origin = params.camera_origin;
        Vector3f ray_direction = params.camera_lower_left + 
                                  u * params.camera_horizontal + 
                                  v * params.camera_vertical - params.camera_origin;
        ray_direction = ray_direction / ray_direction.length();  // Normalize
        
        // Initialize payload
        RayPayload payload;
        payload.color = Vector3f(0.0f, 0.0f, 0.0f);
        payload.attenuation = Vector3f(1.0f, 1.0f, 1.0f);
        payload.origin = ray_origin;
        payload.direction = ray_direction;
        payload.depth = 0;
        payload.done = false;
        payload.seed = seed;
        
        // Trace rays
        while (!payload.done && payload.depth < params.max_depth) {
            uint32_t p0, p1;
            uint64_t ptr = reinterpret_cast<uint64_t>(&payload);
            p0 = static_cast<uint32_t>(ptr);
            p1 = static_cast<uint32_t>(ptr >> 32);
            
            optixTrace(
                reinterpret_cast<OptixTraversableHandle>(params.traversable_handle),
                make_float3(payload.origin.x(), payload.origin.y(), payload.origin.z()),
                make_float3(payload.direction.x(), payload.direction.y(), payload.direction.z()),
                0.001f,                     // tmin
                1e16f,                      // tmax
                0.0f,                       // rayTime
                OptixVisibilityMask(255),   // visibilityMask
                OPTIX_RAY_FLAG_NONE,        // rayFlags
                RAY_TYPE_RADIANCE,          // SBT offset
                RAY_TYPE_COUNT,             // SBT stride
                RAY_TYPE_RADIANCE,          // missSBTIndex
                p0, p1                      // payload
            );
            
            payload.depth++;
        }
        
        pixel_color += payload.color;
        seed = payload.seed;
    }
    
    // Average samples
    pixel_color = pixel_color / static_cast<float>(params.samples_per_pixel);
    
    // Tone mapping (ACES approximation)
    pixel_color = pixel_color / (pixel_color + Vector3f(1.0f, 1.0f, 1.0f));
    
    // Gamma correction
    const float inv_gamma = 1.0f / GAMMA_CORRECTION;
    pixel_color.e[0] = powf(pixel_color.e[0], inv_gamma);
    pixel_color.e[1] = powf(pixel_color.e[1], inv_gamma);
    pixel_color.e[2] = powf(pixel_color.e[2], inv_gamma);
    
    // Write to output
    const int pixel_idx = (y * params.width + x) * 3;
    params.image[pixel_idx + 0] = static_cast<unsigned char>(255.99f * fminf(pixel_color.x(), 1.0f));
    params.image[pixel_idx + 1] = static_cast<unsigned char>(255.99f * fminf(pixel_color.y(), 1.0f));
    params.image[pixel_idx + 2] = static_cast<unsigned char>(255.99f * fminf(pixel_color.z(), 1.0f));
}

// ============================================================================
// MISS PROGRAM
// ============================================================================

extern "C" __global__ void __miss__ms() {
    // Get payload
    uint64_t ptr = static_cast<uint64_t>(optixGetPayload_0()) |
                   (static_cast<uint64_t>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(ptr);
    
    // Background color (black for closed scenes like Cornell Box)
    payload->color += payload->attenuation * Vector3f(0.0f, 0.0f, 0.0f);
    payload->done = true;
}

// ============================================================================
// CLOSEST HIT PROGRAM
// ============================================================================

extern "C" __global__ void __closesthit__ch() {
    // Get payload
    uint64_t ptr = static_cast<uint64_t>(optixGetPayload_0()) |
                   (static_cast<uint64_t>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(ptr);
    
    // Get primitive index
    const int prim_idx = optixGetPrimitiveIndex();
    const Primitive* primitives = reinterpret_cast<const Primitive*>(params.primitives);
    
    if (prim_idx >= 0 && prim_idx < params.num_primitives) {
        const Primitive& prim = primitives[prim_idx];
        
        // Get hit point and normal
        const float t = optixGetRayTmax();
        Vector3f hit_point = payload->origin + t * payload->direction;
        Vector3f normal = prim.getNormal();
        
        // Flip normal if needed
        if (dot(payload->direction, normal) > 0) {
            normal = -normal;
        }
        
        // Add emission
        Vector3f Le = prim.getLe();
        payload->color += payload->attenuation * Le;
        
        // Get BSDF
        Vector3f bsdf = prim.getBSDF();
        
        // Russian roulette
        if (payload->depth > 2) {
            float max_component = fmaxf(bsdf.x(), fmaxf(bsdf.y(), bsdf.z()));
            if (randomFloat(payload->seed) > max_component) {
                payload->done = true;
                return;
            }
            bsdf = bsdf / max_component;
        }
        
        // Update attenuation
        payload->attenuation *= bsdf;
        
        // Check for negligible attenuation
        if (payload->attenuation.length() < 1e-4f) {
            payload->done = true;
            return;
        }
        
        // Sample new direction (cosine-weighted hemisphere)
        float r1 = randomFloat(payload->seed);
        float r2 = randomFloat(payload->seed);
        
        float r = sqrtf(r1);
        float theta = 2.0f * PI * r2;
        float x = r * cosf(theta);
        float y = r * sinf(theta);
        float z = sqrtf(1.0f - r1);
        
        // Build local coordinate system
        Vector3f tangent, bitangent;
        if (fabsf(normal.z()) < NORMAL_EPSILON) {
            tangent = unit_vector(cross(Vector3f(0, 0, 1), normal));
        } else {
            tangent = unit_vector(cross(Vector3f(1, 0, 0), normal));
        }
        bitangent = cross(normal, tangent);
        
        // Transform to world space
        Vector3f new_direction = x * tangent + y * bitangent + z * normal;
        new_direction = new_direction / new_direction.length();
        
        // Update ray
        payload->origin = hit_point + normal * 0.001f;
        payload->direction = new_direction;
    } else {
        payload->done = true;
    }
}

// ============================================================================
// ANY HIT PROGRAM (for shadow rays)
// ============================================================================

extern "C" __global__ void __anyhit__shadow() {
    // Terminate on any hit for shadow rays
    optixTerminateRay();
}

#else // !ENABLE_OPTIX

// Placeholder when OptiX is not enabled
// This file will not be compiled in that case

#endif // ENABLE_OPTIX
