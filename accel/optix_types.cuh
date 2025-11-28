#ifndef OPTIX_TYPES_CUH
#define OPTIX_TYPES_CUH

/**
 * @file optix_types.cuh
 * @brief OptiX type definitions and constants for ray tracing acceleration
 * 
 * This file defines the data structures used for OptiX integration.
 * OptiX provides hardware-accelerated ray tracing on NVIDIA RTX GPUs.
 */

#include "../vector.h"

// ============================================================================
// OPTIX FEATURE FLAGS
// ============================================================================

// Set to 1 to enable OptiX support (requires OptiX SDK)
#ifndef ENABLE_OPTIX
#define ENABLE_OPTIX 0
#endif

// OptiX version requirements
#define OPTIX_MIN_VERSION 70500  // OptiX 7.5

// ============================================================================
// RAY TYPES
// ============================================================================

/**
 * @brief Ray type enumeration for OptiX
 */
enum RayType {
    RAY_TYPE_RADIANCE = 0,      // Primary rays for rendering
    RAY_TYPE_SHADOW = 1,        // Shadow rays for visibility
    RAY_TYPE_COUNT
};

// ============================================================================
// LAUNCH PARAMETERS
// ============================================================================

/**
 * @brief Parameters passed to OptiX ray generation programs
 */
struct OptixLaunchParams {
    // Output image
    unsigned char* image;
    int width;
    int height;
    
    // Camera parameters
    Vector3f camera_origin;
    Vector3f camera_lower_left;
    Vector3f camera_horizontal;
    Vector3f camera_vertical;
    
    // Scene data
    void* traversable_handle;   // OptixTraversableHandle
    void* primitives;           // Device primitives array
    int num_primitives;
    
    // Rendering parameters
    int samples_per_pixel;
    int max_depth;
    unsigned int frame_number;
    
    // Random seed
    unsigned int random_seed;
};

// ============================================================================
// SHADER BINDING TABLE RECORDS
// ============================================================================

/**
 * @brief Header for SBT records
 */
struct alignas(16) SbtRecordHeader {
    char header[32];  // OPTIX_SBT_RECORD_HEADER_SIZE
};

/**
 * @brief Ray generation SBT record
 */
struct RayGenSbtRecord {
    SbtRecordHeader header;
    // No custom data needed
};

/**
 * @brief Miss program SBT record
 */
struct MissSbtRecord {
    SbtRecordHeader header;
    Vector3f background_color;
};

/**
 * @brief Hit group SBT record
 */
struct HitGroupSbtRecord {
    SbtRecordHeader header;
    int primitive_index;
    Vector3f bsdf;
    Vector3f emission;
};

// ============================================================================
// OPTIX STATE STRUCTURE
// ============================================================================

/**
 * @brief OptiX context and pipeline state
 * 
 * This structure holds all OptiX-related state including:
 * - Device context
 * - Pipeline and modules
 * - Shader binding table
 * - Acceleration structures
 */
struct OptixState {
    // OptiX handles (void* for compilation without OptiX SDK)
    void* context;              // OptixDeviceContext
    void* pipeline;             // OptixPipeline
    void* module;               // OptixModule
    
    // Program groups
    void* raygen_prog_group;    // OptixProgramGroup
    void* miss_prog_group;      // OptixProgramGroup
    void* hitgroup_prog_group;  // OptixProgramGroup
    
    // Acceleration structure
    void* gas_handle;           // OptixTraversableHandle
    void* d_gas_output_buffer;  // CUdeviceptr
    
    // Shader binding table
    void* d_raygen_record;      // CUdeviceptr
    void* d_miss_record;        // CUdeviceptr
    void* d_hitgroup_records;   // CUdeviceptr
    
    // Launch parameters
    OptixLaunchParams params;
    void* d_params;             // CUdeviceptr
    
    // State flags
    bool is_initialized;
    bool pipeline_valid;
    
    // Statistics
    size_t gas_memory_usage;
    float build_time_ms;
    
    /**
     * @brief Constructor
     */
    OptixState() 
        : context(nullptr)
        , pipeline(nullptr)
        , module(nullptr)
        , raygen_prog_group(nullptr)
        , miss_prog_group(nullptr)
        , hitgroup_prog_group(nullptr)
        , gas_handle(nullptr)
        , d_gas_output_buffer(nullptr)
        , d_raygen_record(nullptr)
        , d_miss_record(nullptr)
        , d_hitgroup_records(nullptr)
        , d_params(nullptr)
        , is_initialized(false)
        , pipeline_valid(false)
        , gas_memory_usage(0)
        , build_time_ms(0.0f)
    {}
    
    /**
     * @brief Check if OptiX is available and initialized
     * @return true if OptiX is ready
     */
    bool isReady() const { 
        return is_initialized && pipeline_valid; 
    }
};

// ============================================================================
// OPTIX BUILD OPTIONS
// ============================================================================

/**
 * @brief Options for building OptiX acceleration structures
 */
struct OptixBuildOptions {
    bool allow_compaction;      // Allow memory compaction
    bool allow_update;          // Allow AS updates
    bool prefer_fast_trace;     // Prioritize trace performance
    bool prefer_fast_build;     // Prioritize build performance
    
    OptixBuildOptions()
        : allow_compaction(true)
        , allow_update(false)
        , prefer_fast_trace(true)
        , prefer_fast_build(false)
    {}
};

#endif // OPTIX_TYPES_CUH
