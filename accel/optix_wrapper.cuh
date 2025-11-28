#ifndef OPTIX_WRAPPER_CUH
#define OPTIX_WRAPPER_CUH

/**
 * @file optix_wrapper.cuh
 * @brief OptiX initialization and management wrapper
 * 
 * This file provides a high-level interface for OptiX functionality.
 * It handles:
 * - OptiX context initialization
 * - Pipeline creation
 * - Acceleration structure building
 * - Shader binding table setup
 */

#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#include "optix_types.cuh"
#include "../primitive.h"

// ============================================================================
// OPTIX WRAPPER CLASS
// ============================================================================

/**
 * @brief High-level wrapper for OptiX functionality
 */
class OptixWrapper {
public:
    /**
     * @brief Constructor
     */
    OptixWrapper();
    
    /**
     * @brief Destructor
     */
    ~OptixWrapper();
    
    /**
     * @brief Initialize OptiX context and pipeline
     * @return true if initialization succeeded
     */
    bool initialize();
    
    /**
     * @brief Check if OptiX is available on this system
     * @return true if OptiX is available
     */
    static bool isAvailable();
    
    /**
     * @brief Build acceleration structure from primitives
     * @param primitives Array of primitives
     * @param num_primitives Number of primitives
     * @param options Build options
     * @return true if build succeeded
     */
    bool buildAccelerationStructure(const Primitive* primitives, 
                                     int num_primitives,
                                     const OptixBuildOptions& options = OptixBuildOptions());
    
    /**
     * @brief Update acceleration structure (for deformable geometry)
     * @param primitives Updated primitives
     * @param num_primitives Number of primitives
     * @return true if update succeeded
     */
    bool updateAccelerationStructure(const Primitive* primitives, int num_primitives);
    
    /**
     * @brief Launch ray tracing
     * @param width Image width
     * @param height Image height
     * @param d_output Device output buffer
     */
    void launch(int width, int height, unsigned char* d_output);
    
    /**
     * @brief Set camera parameters
     * @param origin Camera origin
     * @param lower_left Lower left corner of view plane
     * @param horizontal Horizontal extent of view plane
     * @param vertical Vertical extent of view plane
     */
    void setCamera(const Vector3f& origin, const Vector3f& lower_left,
                   const Vector3f& horizontal, const Vector3f& vertical);
    
    /**
     * @brief Set rendering parameters
     * @param spp Samples per pixel
     * @param max_depth Maximum ray depth
     */
    void setRenderParams(int spp, int max_depth);
    
    /**
     * @brief Get OptiX state
     * @return Reference to OptiX state
     */
    const OptixState& getState() const { return state_; }
    
    /**
     * @brief Check if ready for rendering
     * @return true if ready
     */
    bool isReady() const { return state_.isReady(); }
    
    /**
     * @brief Get last error message
     * @return Error message string
     */
    const std::string& getLastError() const { return last_error_; }
    
    /**
     * @brief Get acceleration structure memory usage
     * @return Memory usage in bytes
     */
    size_t getASMemoryUsage() const { return state_.gas_memory_usage; }
    
    /**
     * @brief Get last build time
     * @return Build time in milliseconds
     */
    float getLastBuildTime() const { return state_.build_time_ms; }
    
    /**
     * @brief Cleanup all resources
     */
    void cleanup();

private:
    OptixState state_;
    std::string last_error_;
    
    // Internal helper methods
    bool createContext();
    bool createModule();
    bool createProgramGroups();
    bool createPipeline();
    bool createSBT(const Primitive* primitives, int num_primitives);
    
    void setError(const std::string& msg);
};

// ============================================================================
// OPTIX UTILITY FUNCTIONS
// ============================================================================

namespace optix_utils {

/**
 * @brief Check OptiX result and log error
 * @param result OptiX result code
 * @param msg Error context message
 * @return true if successful
 */
bool checkResult(int result, const char* msg);

/**
 * @brief Get OptiX version string
 * @return Version string
 */
std::string getVersionString();

/**
 * @brief Get available GPU memory for acceleration structures
 * @return Available memory in bytes
 */
size_t getAvailableMemory();

/**
 * @brief Print OptiX device capabilities
 */
void printDeviceCapabilities();

} // namespace optix_utils

// ============================================================================
// OPTIX WRAPPER IMPLEMENTATION (Header-only for simplicity)
// ============================================================================

#if ENABLE_OPTIX

// Include actual OptiX headers when enabled
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

// TODO: Full OptiX implementation
// When OptiX SDK is available, implement the following:
// 1. OptixWrapper::initialize() - Create OptiX context and initialize device
// 2. OptixWrapper::createContext() - Set up OptixDeviceContext with logging callback
// 3. OptixWrapper::createModule() - Compile PTX from optix_programs.cu
// 4. OptixWrapper::createProgramGroups() - Create raygen, miss, and hitgroup programs
// 5. OptixWrapper::createPipeline() - Link program groups into pipeline
// 6. OptixWrapper::buildAccelerationStructure() - Build GAS from triangles
// 7. OptixWrapper::createSBT() - Set up shader binding table
// 8. OptixWrapper::launch() - Launch ray tracing with optixLaunch()
//
// Reference: NVIDIA OptiX Programming Guide
// https://raytracing-docs.nvidia.com/optix7/guide/index.html

#else

// Stub implementations when OptiX is not available

inline OptixWrapper::OptixWrapper() {}
inline OptixWrapper::~OptixWrapper() { cleanup(); }

inline bool OptixWrapper::initialize() {
    setError("OptiX is not enabled. Rebuild with ENABLE_OPTIX=1 and OptiX SDK.");
    return false;
}

inline bool OptixWrapper::isAvailable() {
    return false;
}

inline bool OptixWrapper::buildAccelerationStructure(const Primitive*, int, 
                                                      const OptixBuildOptions&) {
    setError("OptiX is not enabled.");
    return false;
}

inline bool OptixWrapper::updateAccelerationStructure(const Primitive*, int) {
    setError("OptiX is not enabled.");
    return false;
}

inline void OptixWrapper::launch(int, int, unsigned char*) {
    // No-op when OptiX is disabled
}

inline void OptixWrapper::setCamera(const Vector3f&, const Vector3f&,
                                     const Vector3f&, const Vector3f&) {
    // No-op when OptiX is disabled
}

inline void OptixWrapper::setRenderParams(int, int) {
    // No-op when OptiX is disabled
}

inline void OptixWrapper::cleanup() {
    state_ = OptixState();
}

inline bool OptixWrapper::createContext() { return false; }
inline bool OptixWrapper::createModule() { return false; }
inline bool OptixWrapper::createProgramGroups() { return false; }
inline bool OptixWrapper::createPipeline() { return false; }
inline bool OptixWrapper::createSBT(const Primitive*, int) { return false; }

inline void OptixWrapper::setError(const std::string& msg) {
    last_error_ = msg;
    std::cerr << "OptiX Error: " << msg << std::endl;
}

namespace optix_utils {

inline bool checkResult(int, const char*) { return false; }
inline std::string getVersionString() { return "OptiX not enabled"; }
inline size_t getAvailableMemory() { return 0; }
inline void printDeviceCapabilities() {
    std::cout << "OptiX is not enabled in this build." << std::endl;
}

} // namespace optix_utils

#endif // ENABLE_OPTIX

#endif // OPTIX_WRAPPER_CUH
