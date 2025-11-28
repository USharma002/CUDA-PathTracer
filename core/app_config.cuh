#ifndef APP_CONFIG_CUH
#define APP_CONFIG_CUH

#include "../vector.h"
#include "../integrator.h"

// ============================================================================
// APPLICATION CONFIGURATION
// ============================================================================

namespace config {

// Default window dimensions
constexpr int DEFAULT_WIDTH = 800;
constexpr int DEFAULT_HEIGHT = 800;

// Camera controls
constexpr float MOUSE_SENSITIVITY = 0.25f;
constexpr float ZOOM_SENSITIVITY = 0.1f;

// Grid visualization - use values from integrator.h to avoid duplication
// GRID_RES and GRID_SIZE are defined in integrator.h
// These aliases provide access in the config namespace
constexpr int CONFIG_GRID_RES = GRID_RES;       // Use GRID_RES from integrator.h
constexpr int CONFIG_GRID_SIZE = GRID_SIZE;     // Use GRID_SIZE from integrator.h

// Radiosity history size (also defined in primitive.h/triangle.h/quad.h)
constexpr int CONFIG_RADIOSITY_HISTORY = RADIOSITY_HISTORY;

// Default paths
constexpr const char* DEFAULT_SCENE_FILE = "./scenes/cbox_quads.obj";

} // namespace config

// ============================================================================
// INTEGRATOR TYPES
// ============================================================================

enum class IntegratorType { 
    PathTracing, 
    Radiosity, 
    RadiosityDelta,
    RadiosityHistory
};

// ============================================================================
// GRID VISUALIZATION MODES
// ============================================================================

enum class GridVisualizationMode { 
    VisibilityCount, 
    RadiosityDistribution 
};

// ============================================================================
// ACCELERATION STRUCTURE TYPES
// ============================================================================

enum class AccelStructType {
    BVH,        // Standard BVH (current implementation)
    OptiX       // OptiX hardware-accelerated BVH
};

// ============================================================================
// APPLICATION CONFIGURATION STRUCTURE
// ============================================================================

struct AppConfig {
    // Rendering settings
    int spp;                        // Samples per pixel
    int max_bounces;                // Maximum ray bounces
    
    // Subdivision settings
    int subdivision_step;
    int radiosity_step;
    
    // History visualization
    int history_step1;
    int history_step2;
    
    // Integrator settings
    IntegratorType current_integrator;
    SamplingMode sampling_mode;
    
    // Acceleration structure
    AccelStructType accel_type;
    
    // Camera settings
    Vector3f camera_origin;
    Vector3f look_at;
    Vector3f up;
    float fov;
    
    // Scene settings
    bool convert_quads_to_triangles;
    
    // OptiX settings (for future use)
    bool use_optix;
    int optix_max_trace_depth;
    
    AppConfig() 
        : spp(1)
        , max_bounces(10)
        , subdivision_step(0)
        , radiosity_step(5)
        , history_step1(0)
        , history_step2(1)
        , current_integrator(IntegratorType::PathTracing)
        , sampling_mode(SAMPLING_BSDF)
        , accel_type(AccelStructType::BVH)
        , camera_origin(0.5f, 3.0f, 8.5f)
        , look_at(0.0f, 2.5f, 0.0f)
        , up(0.0f, 1.0f, 0.0f)
        , fov(40.0f)
        , convert_quads_to_triangles(false)
        , use_optix(false)
        , optix_max_trace_depth(2)
    {}
    
    // Reset to defaults
    void reset() {
        *this = AppConfig();
    }
};

#endif // APP_CONFIG_CUH
