/**
 * @file optix_wrapper.h
 * @brief OptiX 7+ integration for hardware-accelerated ray tracing
 * 
 * This wrapper provides:
 * - Acceleration structure (BVH) building using OptiX
 * - Hardware-accelerated ray-scene intersection
 * - Support for both triangles and quads (quads converted to 2 triangles)
 * - Custom hit data for path tracing (BSDF, radiosity grid, etc.)
 * - Comprehensive logging for debugging and performance analysis
 */

#ifndef OPTIX_WRAPPER_H
#define OPTIX_WRAPPER_H

#include "utils/optix_logger.h"

// Check if OptiX is available
#ifdef USE_OPTIX

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include "core/vector.h"
#include "rendering/primitive.h"
#include "rendering/surface_interaction_record.h"
#include "rendering/render_config.h"

// ============================================================================
// ERROR CHECKING MACROS
// ============================================================================

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            std::stringstream ss;                                             \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
               << cudaGetErrorString(error);                                  \
            OptixLogger::getInstance().error("CUDA", ss.str());               \
            throw std::runtime_error(ss.str());                               \
        }                                                                     \
    } while (0)

#define OPTIX_CHECK(call)                                                     \
    do {                                                                      \
        OptixResult result = call;                                            \
        if (result != OPTIX_SUCCESS) {                                        \
            std::stringstream ss;                                             \
            ss << "OptiX error at " << __FILE__ << ":" << __LINE__ << ": "    \
               << optixGetErrorName(result) << " - "                          \
               << optixGetErrorString(result);                                \
            OptixLogger::getInstance().error("OptiX", ss.str());              \
            throw std::runtime_error(ss.str());                               \
        }                                                                     \
    } while (0)

// ============================================================================
// LAUNCH PARAMETERS - Shared between host and device
// ============================================================================

struct OptixLaunchParams {
    // Output buffer
    unsigned char* image;
    int width;
    int height;
    
    // Camera parameters
    Vector3f cam_origin;
    Vector3f cam_lower_left;
    Vector3f cam_horizontal;
    Vector3f cam_vertical;
    
    // Acceleration structure handle
    OptixTraversableHandle traversable;
    
    // Primitive data for shading (indexed by primitive ID)
    struct PrimitiveData {
        Vector3f bsdf;
        Vector3f Le;
        Vector3f normal;
        Vector3f radiosity;
        float* radiosity_grid;  // Pointer to GRID_SIZE floats (luminance)
        int original_prim_idx;  // Index into original primitive array
    };
    PrimitiveData* primitive_data;
    int num_primitives;
    
    // Full primitive array for advanced operations
    Primitive* primitives;
    
    // Filtered PDFs
    float* filtered_radiosity;
    bool use_filtered;
    
    // Sampling configuration
    int spp;
    int max_depth;
    float mis_bsdf_fraction;
    
    // Random state
    curandState* rand_state;
    
    // Frame counter for progressive rendering
    int frame_number;
};

// ============================================================================
// SBT RECORD TYPES
// ============================================================================

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData {};
struct MissData {
    Vector3f bg_color;
};
struct HitGroupData {
    // Empty - we use primitive ID to index into primitive_data array
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

// ============================================================================
// OPTIX STATISTICS
// ============================================================================

struct OptixStats {
    // Timing
    double init_time_ms;
    double build_time_ms;
    double compact_time_ms;
    double last_trace_time_ms;
    
    // Memory
    size_t gas_size_before_compact;
    size_t gas_size_after_compact;
    size_t total_device_memory;
    
    // Geometry
    int num_input_primitives;
    int num_triangles;
    int num_quads_converted;
    
    // Performance counters
    long long total_rays_traced;
    int frame_count;
    
    void reset() {
        init_time_ms = build_time_ms = compact_time_ms = last_trace_time_ms = 0.0;
        gas_size_before_compact = gas_size_after_compact = total_device_memory = 0;
        num_input_primitives = num_triangles = num_quads_converted = 0;
        total_rays_traced = 0;
        frame_count = 0;
    }
    
    void logSummary() {
        auto& logger = OptixLogger::getInstance();
        std::stringstream ss;
        
        logger.info("OptiX Stats", "========== OptiX Statistics ==========");
        
        ss << "Initialization: " << std::fixed << std::setprecision(2) << init_time_ms << " ms";
        logger.info("OptiX Stats", ss.str()); ss.str("");
        
        ss << "BVH Build: " << build_time_ms << " ms (Compact: " << compact_time_ms << " ms)";
        logger.info("OptiX Stats", ss.str()); ss.str("");
        
        ss << "Geometry: " << num_input_primitives << " primitives -> " 
           << num_triangles << " triangles (" << num_quads_converted << " quads converted)";
        logger.info("OptiX Stats", ss.str()); ss.str("");
        
        ss << "GAS Memory: " << (gas_size_before_compact / 1024.0 / 1024.0) << " MB -> "
           << (gas_size_after_compact / 1024.0 / 1024.0) << " MB ("
           << (100.0 * gas_size_after_compact / gas_size_before_compact) << "%)";
        logger.info("OptiX Stats", ss.str()); ss.str("");
        
        ss << "Total Device Memory: " << (total_device_memory / 1024.0 / 1024.0) << " MB";
        logger.info("OptiX Stats", ss.str()); ss.str("");
        
        if (total_rays_traced > 0 && frame_count > 0) {
            double avg_rays_per_frame = (double)total_rays_traced / frame_count;
            ss << "Rays: " << (total_rays_traced / 1e6) << "M total, "
               << (avg_rays_per_frame / 1e6) << "M/frame avg";
            logger.info("OptiX Stats", ss.str());
        }
        
        logger.info("OptiX Stats", "=======================================");
    }
};

// ============================================================================
// OPTIX CONTEXT MANAGER
// ============================================================================

class OptixWrapper {
public:
    OptixWrapper() : m_context(nullptr), m_pipeline(nullptr), m_gas_handle(0),
                     m_d_gas_buffer(nullptr), m_d_params(nullptr),
                     m_d_primitive_data(nullptr), m_d_vertices(nullptr),
                     m_initialized(false) {
        m_stats.reset();
    }
    
    ~OptixWrapper() {
        cleanup();
    }
    
    /**
     * Initialize OptiX context and pipeline
     */
    bool initialize() {
        if (m_initialized) {
            OPTIX_LOG_DEBUG("Already initialized, skipping");
            return true;
        }
        
        auto& logger = OptixLogger::getInstance();
        logger.info("OptiX Init", "========== OptiX Initialization ==========");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // Initialize CUDA
            logger.debug("OptiX Init", "Initializing CUDA...");
            CUDA_CHECK(cudaFree(0));
            
            int device;
            cudaGetDevice(&device);
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            
            std::stringstream ss;
            ss << "CUDA Device: " << props.name << " (SM " << props.major << "." << props.minor << ")";
            logger.info("OptiX Init", ss.str());
            
            ss.str("");
            ss << "Compute Capability: " << props.major << "." << props.minor 
               << ", Memory: " << (props.totalGlobalMem / 1024 / 1024) << " MB";
            logger.info("OptiX Init", ss.str());
            
            // Check for RT cores
            bool hasRTCores = (props.major >= 7 && props.minor >= 5) || props.major >= 8;
            if (hasRTCores) {
                logger.info("OptiX Init", "RT Cores: Available (RTX GPU detected)");
            } else {
                logger.warning("OptiX Init", "RT Cores: Not available (compute-based ray tracing)");
            }
            
            // Initialize OptiX
            logger.debug("OptiX Init", "Loading OptiX library...");
            OPTIX_CHECK(optixInit());
            logger.info("OptiX Init", "OptiX library loaded successfully");
            
            // Create context
            logger.debug("OptiX Init", "Creating OptiX device context...");
            CUcontext cu_context = 0;
            
            OptixDeviceContextOptions context_options = {};
            context_options.logCallbackFunction = &OptixLogger::optixCallback;
            context_options.logCallbackLevel = 4;  // Print all messages
            
            OPTIX_CHECK(optixDeviceContextCreate(cu_context, &context_options, &m_context));
            logger.info("OptiX Init", "OptiX device context created");
            
            // Query OptiX version
            unsigned int optix_version;
            OPTIX_CHECK(optixDeviceContextGetProperty(m_context, 
                OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH, &optix_version, sizeof(optix_version)));
            
            ss.str("");
            ss << "OptiX Max Trace Depth: " << optix_version;
            logger.info("OptiX Init", ss.str());
            
            auto end = std::chrono::high_resolution_clock::now();
            m_stats.init_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            ss.str("");
            ss << "Initialization complete in " << std::fixed << std::setprecision(2) 
               << m_stats.init_time_ms << " ms";
            logger.info("OptiX Init", ss.str());
            logger.info("OptiX Init", "==========================================");
            
            m_initialized = true;
            return true;
            
        } catch (const std::exception& e) {
            logger.error("OptiX Init", std::string("Initialization failed: ") + e.what());
            return false;
        }
    }
    
    /**
     * Build acceleration structure from primitives
     * Converts quads to triangles for OptiX
     */
    bool buildAccelStructure(Primitive* primitives, int num_primitives) {
        if (!m_initialized) {
            OPTIX_LOG_ERROR("Cannot build accel structure: OptiX not initialized!");
            return false;
        }
        
        auto& logger = OptixLogger::getInstance();
        logger.info("OptiX Build", "========== Building Acceleration Structure ==========");
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        m_stats.num_input_primitives = num_primitives;
        
        std::stringstream ss;
        ss << "Input: " << num_primitives << " primitives";
        logger.info("OptiX Build", ss.str());
        
        // Convert primitives to triangle soup
        logger.debug("OptiX Build", "Converting primitives to triangle mesh...");
        
        std::vector<float3> vertices;
        std::vector<uint3> indices;
        std::vector<OptixLaunchParams::PrimitiveData> prim_data;
        
        int tri_count = 0;
        int quad_count = 0;
        
        for (int i = 0; i < num_primitives; i++) {
            const Primitive& prim = primitives[i];
            
            if (prim.type == PRIM_TRIANGLE) {
                // Add triangle vertices
                int v_start = vertices.size();
                vertices.push_back(make_float3(prim.tri.v0.x(), prim.tri.v0.y(), prim.tri.v0.z()));
                vertices.push_back(make_float3(prim.tri.v1.x(), prim.tri.v1.y(), prim.tri.v1.z()));
                vertices.push_back(make_float3(prim.tri.v2.x(), prim.tri.v2.y(), prim.tri.v2.z()));
                indices.push_back(make_uint3(v_start, v_start + 1, v_start + 2));
                
                // Store primitive data
                OptixLaunchParams::PrimitiveData pd;
                pd.bsdf = prim.tri.bsdf;
                pd.Le = prim.tri.Le;
                pd.normal = prim.tri.normal;
                pd.radiosity = prim.tri.radiosity;
                pd.original_prim_idx = i;
                prim_data.push_back(pd);
                tri_count++;
                
            } else {  // PRIM_QUAD - convert to 2 triangles
                quad_count++;
                
                // Triangle 1: v00, v10, v11
                int v_start = vertices.size();
                vertices.push_back(make_float3(prim.quad.v00.x(), prim.quad.v00.y(), prim.quad.v00.z()));
                vertices.push_back(make_float3(prim.quad.v10.x(), prim.quad.v10.y(), prim.quad.v10.z()));
                vertices.push_back(make_float3(prim.quad.v11.x(), prim.quad.v11.y(), prim.quad.v11.z()));
                indices.push_back(make_uint3(v_start, v_start + 1, v_start + 2));
                
                OptixLaunchParams::PrimitiveData pd1;
                pd1.bsdf = prim.quad.bsdf;
                pd1.Le = prim.quad.Le;
                pd1.normal = prim.quad.normal;
                pd1.radiosity = prim.quad.radiosity;
                pd1.original_prim_idx = i;
                prim_data.push_back(pd1);
                tri_count++;
                
                // Triangle 2: v00, v11, v01
                v_start = vertices.size();
                vertices.push_back(make_float3(prim.quad.v00.x(), prim.quad.v00.y(), prim.quad.v00.z()));
                vertices.push_back(make_float3(prim.quad.v11.x(), prim.quad.v11.y(), prim.quad.v11.z()));
                vertices.push_back(make_float3(prim.quad.v01.x(), prim.quad.v01.y(), prim.quad.v01.z()));
                indices.push_back(make_uint3(v_start, v_start + 1, v_start + 2));
                
                OptixLaunchParams::PrimitiveData pd2;
                pd2.bsdf = prim.quad.bsdf;
                pd2.Le = prim.quad.Le;
                pd2.normal = prim.quad.normal;
                pd2.radiosity = prim.quad.radiosity;
                pd2.original_prim_idx = i;  // Same original index for both triangles
                prim_data.push_back(pd2);
                tri_count++;
            }
        }
        
        m_stats.num_triangles = tri_count;
        m_stats.num_quads_converted = quad_count;
        
        ss.str("");
        ss << "Converted: " << tri_count << " triangles (" << quad_count << " quads -> " 
           << (quad_count * 2) << " triangles)";
        logger.info("OptiX Build", ss.str());
        
        // Upload vertices to GPU
        logger.debug("OptiX Build", "Uploading geometry to GPU...");
        
        size_t vertices_size = vertices.size() * sizeof(float3);
        CUDA_CHECK(cudaMalloc(&m_d_vertices, vertices_size));
        CUDA_CHECK(cudaMemcpy(m_d_vertices, vertices.data(), vertices_size, cudaMemcpyHostToDevice));
        logger.logMemoryAlloc("vertices", vertices_size);
        m_stats.total_device_memory += vertices_size;
        
        // Upload indices to GPU
        size_t indices_size = indices.size() * sizeof(uint3);
        CUdeviceptr d_indices;
        CUDA_CHECK(cudaMalloc((void**)&d_indices, indices_size));
        CUDA_CHECK(cudaMemcpy((void*)d_indices, indices.data(), indices_size, cudaMemcpyHostToDevice));
        logger.logMemoryAlloc("indices", indices_size);
        m_stats.total_device_memory += indices_size;
        
        // Upload primitive data to GPU
        size_t prim_data_size = prim_data.size() * sizeof(OptixLaunchParams::PrimitiveData);
        CUDA_CHECK(cudaMalloc(&m_d_primitive_data, prim_data_size));
        CUDA_CHECK(cudaMemcpy(m_d_primitive_data, prim_data.data(), prim_data_size, cudaMemcpyHostToDevice));
        logger.logMemoryAlloc("primitive_data", prim_data_size);
        m_stats.total_device_memory += prim_data_size;
        m_num_optix_triangles = tri_count;
        
        // Build OptiX acceleration structure
        logger.info("OptiX Build", "Building OptiX GAS (Geometry Acceleration Structure)...");
        
        auto build_start = std::chrono::high_resolution_clock::now();
        
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | 
                                   OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        
        // Triangle input
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        
        CUdeviceptr d_vertices_ptr = (CUdeviceptr)m_d_vertices;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
        triangle_input.triangleArray.numVertices = vertices.size();
        triangle_input.triangleArray.vertexBuffers = &d_vertices_ptr;
        
        triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.indexStrideInBytes = sizeof(uint3);
        triangle_input.triangleArray.numIndexTriplets = indices.size();
        triangle_input.triangleArray.indexBuffer = d_indices;
        
        // All triangles use the same SBT record
        uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;
        
        // Query memory requirements
        logger.debug("OptiX Build", "Computing memory requirements...");
        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            m_context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));
        
        ss.str("");
        ss << "GAS Requirements - Temp: " << (gas_buffer_sizes.tempSizeInBytes / 1024.0) << " KB, "
           << "Output: " << (gas_buffer_sizes.outputSizeInBytes / 1024.0) << " KB";
        logger.debug("OptiX Build", ss.str());
        
        // Allocate temporary buffer
        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer, gas_buffer_sizes.tempSizeInBytes));
        logger.logMemoryAlloc("temp_buffer", gas_buffer_sizes.tempSizeInBytes);
        
        // Allocate output buffer (non-compacted)
        CUdeviceptr d_buffer_temp_output;
        CUDA_CHECK(cudaMalloc((void**)&d_buffer_temp_output, gas_buffer_sizes.outputSizeInBytes));
        logger.logMemoryAlloc("gas_output", gas_buffer_sizes.outputSizeInBytes);
        m_stats.gas_size_before_compact = gas_buffer_sizes.outputSizeInBytes;
        
        // Allocate compacted size output
        CUdeviceptr d_compacted_size;
        CUDA_CHECK(cudaMalloc((void**)&d_compacted_size, sizeof(uint64_t)));
        
        OptixAccelEmitDesc emit_desc;
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = d_compacted_size;
        
        // Build acceleration structure
        logger.debug("OptiX Build", "Executing optixAccelBuild...");
        OPTIX_CHECK(optixAccelBuild(
            m_context, 0, &accel_options,
            &triangle_input, 1,
            d_temp_buffer, gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output, gas_buffer_sizes.outputSizeInBytes,
            &m_gas_handle, &emit_desc, 1));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto build_end = std::chrono::high_resolution_clock::now();
        m_stats.build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
        
        ss.str("");
        ss << "GAS build completed in " << std::fixed << std::setprecision(2) 
           << m_stats.build_time_ms << " ms";
        logger.info("OptiX Build", ss.str());
        
        // Get compacted size
        uint64_t compacted_size;
        CUDA_CHECK(cudaMemcpy(&compacted_size, (void*)d_compacted_size, 
                             sizeof(uint64_t), cudaMemcpyDeviceToHost));
        
        // Compact if beneficial
        auto compact_start = std::chrono::high_resolution_clock::now();
        
        if (compacted_size < gas_buffer_sizes.outputSizeInBytes) {
            logger.debug("OptiX Build", "Compacting GAS...");
            CUDA_CHECK(cudaMalloc((void**)&m_d_gas_buffer, compacted_size));
            OPTIX_CHECK(optixAccelCompact(m_context, 0, m_gas_handle,
                                          (CUdeviceptr)m_d_gas_buffer, compacted_size,
                                          &m_gas_handle));
            CUDA_CHECK(cudaFree((void*)d_buffer_temp_output));
            
            m_stats.gas_size_after_compact = compacted_size;
            m_stats.total_device_memory += compacted_size;
            
            ss.str("");
            ss << "GAS compacted: " << (gas_buffer_sizes.outputSizeInBytes / 1024.0) << " KB -> " 
               << (compacted_size / 1024.0) << " KB (" 
               << std::fixed << std::setprecision(1) 
               << (100.0 - 100.0 * compacted_size / gas_buffer_sizes.outputSizeInBytes) << "% reduction)";
            logger.info("OptiX Build", ss.str());
        } else {
            m_d_gas_buffer = (void*)d_buffer_temp_output;
            m_stats.gas_size_after_compact = gas_buffer_sizes.outputSizeInBytes;
            m_stats.total_device_memory += gas_buffer_sizes.outputSizeInBytes;
            logger.debug("OptiX Build", "Compaction not beneficial, using original GAS");
        }
        
        auto compact_end = std::chrono::high_resolution_clock::now();
        m_stats.compact_time_ms = std::chrono::duration<double, std::milli>(compact_end - compact_start).count();
        
        // Cleanup temporary buffers
        CUDA_CHECK(cudaFree((void*)d_temp_buffer));
        logger.logMemoryFree("temp_buffer");
        CUDA_CHECK(cudaFree((void*)d_compacted_size));
        CUDA_CHECK(cudaFree((void*)d_indices));
        logger.logMemoryFree("indices");
        
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        
        ss.str("");
        ss << "Total build time: " << std::fixed << std::setprecision(2) << total_ms << " ms";
        logger.info("OptiX Build", ss.str());
        logger.info("OptiX Build", "====================================================");
        
        return true;
    }
    
    /**
     * Get the traversable handle for ray tracing
     */
    OptixTraversableHandle getTraversable() const { return m_gas_handle; }
    
    /**
     * Get primitive data array
     */
    OptixLaunchParams::PrimitiveData* getPrimitiveData() const { 
        return (OptixLaunchParams::PrimitiveData*)m_d_primitive_data; 
    }
    
    /**
     * Get number of OptiX triangles
     */
    int getNumTriangles() const { return m_num_optix_triangles; }
    
    /**
     * Check if OptiX is available and initialized
     */
    bool isAvailable() const { return m_initialized; }
    
    /**
     * Get statistics
     */
    const OptixStats& getStats() const { return m_stats; }
    OptixStats& getStats() { return m_stats; }
    
    /**
     * Log current statistics summary
     */
    void logStats() const { m_stats.logSummary(); }
    
    /**
     * Cleanup resources
     */
    void cleanup() {
        auto& logger = OptixLogger::getInstance();
        
        if (!m_initialized) return;
        
        logger.info("OptiX Cleanup", "Cleaning up OptiX resources...");
        
        if (m_d_gas_buffer) { 
            cudaFree(m_d_gas_buffer); 
            logger.logMemoryFree("gas_buffer");
            m_d_gas_buffer = nullptr; 
        }
        if (m_d_primitive_data) { 
            cudaFree(m_d_primitive_data);
            logger.logMemoryFree("primitive_data");
            m_d_primitive_data = nullptr; 
        }
        if (m_d_vertices) { 
            cudaFree(m_d_vertices);
            logger.logMemoryFree("vertices");
            m_d_vertices = nullptr; 
        }
        if (m_d_params) { 
            cudaFree(m_d_params);
            logger.logMemoryFree("params");
            m_d_params = nullptr; 
        }
        
        if (m_pipeline) { 
            optixPipelineDestroy(m_pipeline);
            logger.debug("OptiX Cleanup", "Pipeline destroyed");
            m_pipeline = nullptr; 
        }
        if (m_context) { 
            optixDeviceContextDestroy(m_context);
            logger.debug("OptiX Cleanup", "Context destroyed");
            m_context = nullptr; 
        }
        
        m_initialized = false;
        m_gas_handle = 0;
        
        logger.info("OptiX Cleanup", "Cleanup complete");
    }
    
private:
    // OptiX context and pipeline
    OptixDeviceContext m_context;
    OptixPipeline m_pipeline;
    OptixTraversableHandle m_gas_handle;
    
    // Device memory
    void* m_d_gas_buffer;
    void* m_d_params;
    void* m_d_primitive_data;
    void* m_d_vertices;
    
    int m_num_optix_triangles;
    bool m_initialized;
    
    // Statistics
    mutable OptixStats m_stats;
};

#else // USE_OPTIX not defined

// ============================================================================
// STUB CLASS WHEN OPTIX IS NOT AVAILABLE
// ============================================================================

class OptixWrapper {
public:
    bool initialize() { 
        OptixLogger::getInstance().warning("OptiX", "OptiX not available - compiled without USE_OPTIX");
        OptixLogger::getInstance().info("OptiX", "Using software BVH for ray tracing");
        return false; 
    }
    bool buildAccelStructure(Primitive*, int) { return false; }
    bool isAvailable() const { return false; }
    void cleanup() {}
    void logStats() const {
        OptixLogger::getInstance().info("OptiX Stats", "OptiX not available - no stats to report");
    }
};

#endif // USE_OPTIX

#endif // OPTIX_WRAPPER_H
