#ifndef RENDER_STATE_CUH
#define RENDER_STATE_CUH

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../vector.h"
#include "../sensor.h"
#include "cuda_utils.cuh"
#include "app_config.cuh"

// ============================================================================
// RENDER STATE MANAGEMENT
// ============================================================================

/**
 * @brief Manages rendering state including image buffers, camera, and random states
 */
struct RenderState {
    // Image buffers
    unsigned char* d_image;         // Device image buffer
    unsigned char* d_prev_image;    // Previous frame for temporal accumulation
    unsigned char* h_image;         // Host image buffer
    
    // OpenGL texture
    GLuint texture;
    
    // Resolution
    int width;
    int height;
    int img_size;                   // width * height * 3
    
    // CUDA execution configuration
    dim3 block;
    dim3 grid;
    
    // Camera
    Sensor h_camera;                // Host camera
    Sensor* d_camera;               // Device camera
    
    // Random number generator states
    curandState* d_rand_state;
    
    // Frame accumulation
    int accumulated_frames;
    bool needs_reset;
    
    /**
     * @brief Constructor with default initialization
     */
    RenderState() 
        : d_image(nullptr)
        , d_prev_image(nullptr)
        , h_image(nullptr)
        , texture(0)
        , width(config::DEFAULT_WIDTH)
        , height(config::DEFAULT_HEIGHT)
        , img_size(0)
        , block(16, 16)
        , grid(1, 1)
        , h_camera(Vector3f(0.5f, 3.0f, 8.5f), Vector3f(0.0f, 2.5f, 0.0f), 
                   Vector3f(0.0f, 1.0f, 0.0f), 40.0f, 1.0f)
        , d_camera(nullptr)
        , d_rand_state(nullptr)
        , accumulated_frames(0)
        , needs_reset(true)
    {}
    
    /**
     * @brief Allocate GPU and host buffers
     */
    void allocateBuffers();
    
    /**
     * @brief Update resolution and reallocate buffers
     * @param w New width
     * @param h New height
     */
    void updateResolution(int w, int h);
    
    /**
     * @brief Update camera state on device
     */
    void updateCamera();
    
    /**
     * @brief Reset accumulated frames
     */
    void resetAccumulation();
    
    /**
     * @brief Cleanup all resources
     */
    void cleanup();
    
    /**
     * @brief Check if rendering needs reset
     * @return true if rendering needs reset
     */
    bool needsReset() const { return needs_reset; }
    
    /**
     * @brief Get current frame count
     * @return Number of accumulated frames
     */
    int getFrameCount() const { return accumulated_frames; }
};

// ============================================================================
// RENDER INITIALIZATION KERNEL
// ============================================================================

/**
 * @brief Initialize random states for rendering
 */
__global__ void render_init(int width, int height, curandState* rand_state);

#endif // RENDER_STATE_CUH
