/**
 * @file application_state.h
 * @brief Application state structures and their implementations
 * 
 * Contains all state structures for rendering, scene management, radiosity,
 * UI, and application configuration.
 */

#ifndef APPLICATION_STATE_H
#define APPLICATION_STATE_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "utils/cuda_utils.h"
#include "core/vector.h"
#include "rendering/sensor.h"
#include "rendering/primitive.h"
#include "rendering/scene.h"
#include "rendering/bvh.h"
#include "rendering/render_config.h"
#include "rendering/ray_tracing_backend.h"
#include "utils/file_manager.h"
#include "rendering/form_factors.h"
#include "rendering/integrator.h"

// Include PBRT loader if enabled
#ifdef USE_PBRT_LOADER
#include "utils/pbrt_loader.h"
#endif

// ============================================================================
// CONSTANTS & ENUMS
// ============================================================================

constexpr int   DEFAULT_WIDTH       = 800;
constexpr int   DEFAULT_HEIGHT      = 800;
constexpr float MOUSE_SENSITIVITY   = 0.25f;
constexpr float ZOOM_SENSITIVITY    = 0.1f;

#define RADIOSITY_HISTORY 10

enum class IntegratorType {
    PathTracing,
    Radiosity
};

enum class GridVisualizationMode { 
    VisibilityCount, 
    RadiosityDistribution 
};

enum class HistogramSamplingSource {
    FORM_FACTOR = 0,
    RADIOSITY = 1,
    BSDF_WEIGHTED = 2,
    MIS = 3
};

// ============================================================================
// FORWARD DECLARATIONS (for kernel functions)
// ============================================================================

__global__ void render_init(int width, int height, curandState* rand_state);

// ============================================================================
// RENDER STATE
// ============================================================================

struct RenderState {
    unsigned char *d_image, *d_prev_image, *h_image;
    GLuint texture;
    int width, height, img_size;
    dim3 block, grid;
    
    Sensor h_camera, *d_camera;
    curandState* d_rand_state;
    
    RenderState() : d_image(nullptr), d_prev_image(nullptr), h_image(nullptr),
                    texture(0), width(DEFAULT_WIDTH), height(DEFAULT_HEIGHT), img_size(0),
                    block(16, 16), grid(1, 1),
                    h_camera(Vector3f(0.5, 3, 8.5), Vector3f(0, 2.5, 0), Vector3f(0, 1, 0), 40.0f, 1.0f),
                    d_camera(nullptr), d_rand_state(nullptr) {}
    
    void allocateBuffers() {
        // Cleanup old buffers
        if (d_image) cudaFree(d_image);
        if (d_prev_image) cudaFree(d_prev_image);
        if (h_image) delete[] h_image;
        if (d_rand_state) cudaFree(d_rand_state);
        
        // Allocate new buffers
        img_size = width * height * 3;
        h_image = new unsigned char[img_size];
        cudaMallocSafe(&d_image, img_size * sizeof(unsigned char), "d_image");
        cudaMallocSafe(&d_prev_image, img_size * sizeof(unsigned char), "d_prev_image");
        cudaMemset(d_prev_image, 0, img_size * sizeof(unsigned char));
        
        // Update camera
        h_camera.image_width = width;
        h_camera.image_height = height;
        h_camera.aspect = (float)width / (float)height;
        h_camera.updateCamera();
        cudaMemcpy(d_camera, &h_camera, sizeof(Sensor), cudaMemcpyHostToDevice);
        
        // Update CUDA grid
        grid = dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        // Update OpenGL texture
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        
        // Initialize random states
        cudaMallocSafe(&d_rand_state, (size_t)width * height * sizeof(curandState), "d_rand_state");
        render_init<<<grid, block>>>(width, height, d_rand_state);
        cudaDeviceSynchronize();
    }
    
    void updateResolution(int w, int h) {
        width = w;
        height = h;
        allocateBuffers();
    }
    
    void cleanup() {
        if (d_image) cudaFree(d_image);
        if (d_prev_image) cudaFree(d_prev_image);
        if (h_image) delete[] h_image;
        if (d_camera) cudaFree(d_camera);
        if (d_rand_state) cudaFree(d_rand_state);
        if (texture) glDeleteTextures(1, &texture);
        
        d_image = d_prev_image = h_image = nullptr;
        d_camera = nullptr;
        d_rand_state = nullptr;
        texture = 0;
    }
};

// ============================================================================
// SCENE STATE
// ============================================================================

// Forward declare convertQuadsToTriangles (implemented at bottom of file)
std::vector<Primitive> convertQuadsToTriangles(const std::vector<Primitive>& primitives);

struct SceneState {
    Primitive *h_primitives, *d_primitives;
    BVHNode* d_bvh_nodes;
    int* d_bvh_indices;
    int bvh_node_count;
    Scene* d_scene;
    Scene h_scene;
    float* d_filtered_formfactor;
    float* d_filtered_radiosity;
    PrecomputedCDF* h_precomputed_cdfs;
    PrecomputedCDF* d_precomputed_cdfs;
    int num_primitives;
    std::string scene_file;
    
    // Reference to config for quad conversion flag
    bool* convert_quads_ptr;
    
    SceneState() : h_primitives(nullptr), d_primitives(nullptr),
                   d_bvh_nodes(nullptr), d_bvh_indices(nullptr), bvh_node_count(0),
                   d_scene(nullptr), d_filtered_formfactor(nullptr), d_filtered_radiosity(nullptr),
                   h_precomputed_cdfs(nullptr), d_precomputed_cdfs(nullptr), num_primitives(0),
                   scene_file("./scenes/cbox_quads.obj"), convert_quads_ptr(nullptr) {}
    
    void loadScene(const std::string& filename, int subdivision_count, bool convert_quads);
    void cleanup();
    void precomputeCDFs();
    void precomputeCDFsFromFiltered();
    
    void setUseFiltered(bool use_filtered) {
        h_scene.use_filtered = use_filtered;
        if (d_scene) {
            cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);
        }
    }
    
    void setMisBsdfFraction(float fraction) {
        h_scene.mis_bsdf_fraction = fraction;
        if (d_scene) {
            cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);
        }
    }
};

// ============================================================================
// RADIOSITY STATE
// ============================================================================

struct RadiosityState {
    Primitive* d_radiosity_primitives;
    float* d_form_factors;
    curandState* d_rand_states;
    int num_iterations, mc_samples;
    bool use_monte_carlo, is_calculated;
    
    RadiosityState() : d_radiosity_primitives(nullptr), d_form_factors(nullptr),
                       d_rand_states(nullptr), num_iterations(10), mc_samples(64),
                       use_monte_carlo(true), is_calculated(false) {}
    
    void runSolver(Primitive* h_prims, int num_prims, BVHNode* d_bvh, int* d_indices,
                   bool enable_filtering, bool use_bilateral, float filter_sigma_spatial, float filter_sigma_range);
    void cleanup();
};

// ============================================================================
// UI STATE
// ============================================================================

struct UIState {
    bool is_dragging, show_grid_window;
    double last_mouse_x, last_mouse_y;
    int hovered_primitive_idx;
    float* h_grid_data;
    GridVisualizationMode grid_viz_mode;
    
    int grid_resolution;
    int top_k_count;
    HistogramSamplingSource histogram_source;
    bool show_top_k_only;
    bool visualize_luminance;
    
    int* top_k_indices;
    float* top_k_values;
    int actual_top_k_count;
    
    UIState() : is_dragging(false), show_grid_window(false),
                last_mouse_x(0.0), last_mouse_y(0.0),
                hovered_primitive_idx(-1), h_grid_data(nullptr),
                grid_viz_mode(GridVisualizationMode::VisibilityCount),
                grid_resolution(GRID_RES), top_k_count(0),
                histogram_source(HistogramSamplingSource::FORM_FACTOR),
                show_top_k_only(false), visualize_luminance(true),
                top_k_indices(nullptr), top_k_values(nullptr), actual_top_k_count(0) {}
    
    void cleanup() {
        if (h_grid_data) delete[] h_grid_data;
        if (top_k_indices) delete[] top_k_indices;
        if (top_k_values) delete[] top_k_values;
        
        h_grid_data = nullptr;
        top_k_indices = nullptr;
        top_k_values = nullptr;
        actual_top_k_count = 0;
    }
};

// ============================================================================
// APPLICATION CONFIG
// ============================================================================

struct AppConfig {
    int spp, subdivision_step, radiosity_step;
    int history_step1, history_step2;
    IntegratorType current_integrator;
    Vector3f camera_origin, look_at, up;
    SamplingMode sampling_mode;
    HistogramSamplingSource histogram_source;
    float fov;
    bool convert_quads_to_triangles;
    int top_k_samples;
    int current_grid_res;
    
    bool enable_grid_filtering;
    bool use_bilateral_filter;
    float filter_sigma_spatial;
    float filter_sigma_range;
    
    float mis_bsdf_fraction;
    bool enable_kernel_profiling;
    
    AppConfig() : spp(1), subdivision_step(0), radiosity_step(5),
                  history_step1(0), history_step2(1),
                  current_integrator(IntegratorType::PathTracing),
                  camera_origin(0.5, 3, 8.5), look_at(0, 2.5, 0),
                  up(0, 1, 0), fov(40.0f), sampling_mode(SamplingMode::SAMPLING_BSDF),
                  histogram_source(HistogramSamplingSource::FORM_FACTOR),
                  convert_quads_to_triangles(false), top_k_samples(0),
                  current_grid_res(DEFAULT_GRID_RES),
                  enable_grid_filtering(false), use_bilateral_filter(true),
                  filter_sigma_spatial(1.5f), filter_sigma_range(0.3f),
                  mis_bsdf_fraction(0.5f), enable_kernel_profiling(true) {}
};

// ============================================================================
// COMBINED APPLICATION STATE
// ============================================================================

struct ApplicationState {
    RenderState render;
    SceneState scene;
    RadiosityState radiosity;
    UIState ui;
    AppConfig config;
};

// Global state instance (declared extern, defined in main.cu)
extern ApplicationState g_state;

// ============================================================================
// KERNEL PROFILE DATA (if profiling enabled)
// ============================================================================

#if ENABLE_KERNEL_PROFILING
extern KernelProfileData* d_kernel_profile;
extern KernelProfileData h_kernel_profile;
#endif

// ============================================================================
// SCENE STATE IMPLEMENTATION
// ============================================================================

inline std::vector<Primitive> convertQuadsToTriangles(const std::vector<Primitive>& primitives) {
    std::vector<Primitive> triangulated;
    triangulated.reserve(primitives.size() * 2);
    
    int quad_count = 0, tri_count = 0;
    
    for (const auto& prim : primitives) {
        if (prim.type == PRIM_QUAD) {
            quad_count++;
            const Quad& q = prim.quad;
            
            Primitive tri1_prim;
            tri1_prim.type = PRIM_TRIANGLE;
            new (&tri1_prim.tri) Triangle(q.v00, q.v10, q.v11, q.bsdf);
            tri1_prim.tri.Le = q.Le;
            tri1_prim.tri.radiosity = q.radiosity;
            tri1_prim.tri.unshot_rad = q.unshot_rad;
            memcpy(tri1_prim.tri.grid, q.grid, sizeof(q.grid));
            memcpy(tri1_prim.tri.radiosity_grid, q.radiosity_grid, sizeof(q.radiosity_grid));
            triangulated.push_back(tri1_prim);
            
            Primitive tri2_prim;
            tri2_prim.type = PRIM_TRIANGLE;
            new (&tri2_prim.tri) Triangle(q.v00, q.v11, q.v01, q.bsdf);
            tri2_prim.tri.Le = q.Le;
            tri2_prim.tri.radiosity = q.radiosity;
            tri2_prim.tri.unshot_rad = q.unshot_rad;
            memcpy(tri2_prim.tri.grid, q.grid, sizeof(q.grid));
            memcpy(tri2_prim.tri.radiosity_grid, q.radiosity_grid, sizeof(q.radiosity_grid));
            triangulated.push_back(tri2_prim);
        } else {
            tri_count++;
            triangulated.push_back(prim);
        }
    }
    
    if (quad_count > 0) {
        std::cout << "  Converted " << quad_count << " quads -> " << (quad_count * 2) 
                  << " triangles (" << tri_count << " triangles unchanged)" << std::endl;
    }
    
    return triangulated;
}

inline void SceneState::loadScene(const std::string& filename, int subdivision_count, bool convert_quads) {
    std::cout << "\n========== LOADING SCENE ==========\n"
              << "File: " << filename << "\n"
              << "Subdivision: " << subdivision_count << std::endl;
    
    cleanup();
    
    Primitive* temp_primitives = nullptr;
    int temp_num_primitives = 0;
    
    // Detect file format and use appropriate loader
    bool load_success = false;
    std::string ext = filename.substr(filename.find_last_of('.'));
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".obj") {
        load_success = loadOBJ(filename, &temp_primitives, temp_num_primitives);
    }
#ifdef USE_PBRT_LOADER
    else if (ext == ".pbrt") {
        load_success = loadPBRT(filename, &temp_primitives, temp_num_primitives);
    }
#endif
    else {
        std::cerr << "ERROR: Unsupported file format: " << ext << std::endl;
    }
    
    if (!load_success) {
        std::cerr << "ERROR: Failed to load scene: " << filename << std::endl;
        return;
    }
    
    std::cout << "Loaded " << temp_num_primitives << " primitives" << std::endl;
    
    std::vector<Primitive> primitives(temp_primitives, temp_primitives + temp_num_primitives);
    delete[] temp_primitives;

    if (convert_quads) {
        std::cout << "Converting quads to triangles..." << std::endl;
        primitives = convertQuadsToTriangles(primitives);
    }
    
    if (subdivision_count > 0) {
        std::cout << "Subdividing..." << std::endl;
        primitives = subdivide_primitives(primitives, subdivision_count);
        std::cout << "After subdivision: " << primitives.size() << " primitives" << std::endl;
    }
    
    num_primitives = primitives.size();
    h_primitives = new Primitive[num_primitives];
    std::copy(primitives.begin(), primitives.end(), h_primitives);
    
    cudaMallocSafe(&d_primitives, num_primitives * sizeof(Primitive), "d_primitives");
    cudaMemcpy(d_primitives, h_primitives, num_primitives * sizeof(Primitive), cudaMemcpyHostToDevice);
    
    size_t pdf_elems = (size_t)num_primitives * GRID_SIZE;
    size_t pdf_bytes = pdf_elems * sizeof(float);
    cudaMallocSafe(&d_filtered_formfactor, pdf_bytes, "d_filtered_formfactor");
    cudaMallocSafe(&d_filtered_radiosity, pdf_bytes, "d_filtered_radiosity");
    cudaMemset(d_filtered_formfactor, 0, pdf_bytes);
    cudaMemset(d_filtered_radiosity, 0, pdf_bytes);
    
    std::cout << "Building BVH..." << std::endl;
    auto bvh_start = std::chrono::high_resolution_clock::now();
    
    // Initialize the ray tracing backend (OptiX if available, software BVH as fallback)
    auto& rtManager = RayTracingManager::getInstance();
    rtManager.initialize(true);  // prefer OptiX
    
    // Build acceleration structures through the manager
    rtManager.buildAccelStructure(h_primitives, num_primitives, 
                                   &d_bvh_nodes, &d_bvh_indices, bvh_node_count);
    
    auto bvh_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(bvh_end - bvh_start);
    std::cout << "BVH built in " << duration.count() << " ms" << std::endl;
    std::cout << "Backend: " << rtManager.getBackendName() << std::endl;
    
    // Log OptiX stats if available
    rtManager.logStats();
    
    h_scene = Scene(d_primitives, num_primitives, d_bvh_nodes, d_bvh_indices);
    h_scene.filtered_formfactor = d_filtered_formfactor;
    h_scene.filtered_radiosity = d_filtered_radiosity;
    cudaMallocSafe(&d_scene, sizeof(Scene), "d_scene");
    cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);
    
    scene_file = filename;
    
    // Estimate memory usage
    size_t bvh_memory = bvh_node_count * sizeof(BVHNode) + num_primitives * sizeof(int);
    
    std::cout << "========== SCENE LOADED SUCCESSFULLY ==========\n"
              << "Total primitives: " << num_primitives << "\n"
              << "BVH nodes: " << bvh_node_count << "\n"
              << "Memory: " << (num_primitives * sizeof(Primitive) + bvh_memory) / 1024.0 
              << " KB\n" << std::endl;
}

inline void SceneState::cleanup() {
    // Cleanup ray tracing backend (OptiX resources if used)
    RayTracingManager::getInstance().cleanup();
    
    if (h_primitives) delete[] h_primitives;
    if (d_primitives) cudaFree(d_primitives);
    if (d_bvh_nodes) cudaFree(d_bvh_nodes);
    if (d_bvh_indices) cudaFree(d_bvh_indices);
    if (d_scene) cudaFree(d_scene);
    if (d_filtered_formfactor) cudaFree(d_filtered_formfactor);
    if (d_filtered_radiosity) cudaFree(d_filtered_radiosity);
    if (h_precomputed_cdfs) delete[] h_precomputed_cdfs;
    if (d_precomputed_cdfs) cudaFree(d_precomputed_cdfs);
    
    h_primitives = d_primitives = nullptr;
    d_bvh_nodes = nullptr;
    d_bvh_indices = nullptr;
    d_scene = nullptr;
    d_filtered_formfactor = nullptr;
    d_filtered_radiosity = nullptr;
    h_precomputed_cdfs = nullptr;
    d_precomputed_cdfs = nullptr;
    num_primitives = 0;
    bvh_node_count = 0;
}

inline void SceneState::precomputeCDFs() {
    if (num_primitives == 0 || h_primitives == nullptr) return;
    
    std::cout << "\n[CDF] Pre-computing CDFs for " << num_primitives << " primitives..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    if (!h_precomputed_cdfs) {
        h_precomputed_cdfs = new PrecomputedCDF[num_primitives];
    }
    
    if (!d_precomputed_cdfs) {
        cudaMalloc(&d_precomputed_cdfs, num_primitives * sizeof(PrecomputedCDF));
    }
    
    int progress_step = (num_primitives > 100) ? num_primitives / 10 : 100;
    
    #pragma omp parallel for schedule(dynamic) if(num_primitives > 100)
    for (int p = 0; p < num_primitives; p++) {
        PrecomputedCDF& cdf = h_precomputed_cdfs[p];
        
        const Vector3f* rad_grid = (h_primitives[p].type == PRIM_TRIANGLE) 
            ? h_primitives[p].tri.radiosity_grid 
            : h_primitives[p].quad.radiosity_grid;
        
        for (int i = 0; i < GRID_SIZE; i++) {
            cdf.pdf[i] = 0.2126f * rad_grid[i].x() + 
                         0.7152f * rad_grid[i].y() + 
                         0.0722f * rad_grid[i].z();
        }
        
        cdf.total_weight = 0.0f;
        for (int v = 0; v < GRID_HALF_RES; v++) {
            float row_sum = 0.0f;
            const int row_offset = v * GRID_RES;
            for (int u = 0; u < GRID_RES; u++) {
                row_sum += cdf.pdf[row_offset + u];
            }
            cdf.row_sums[v] = row_sum;
            cdf.total_weight += row_sum;
        }
        
        float running = 0.0f;
        float inv_total = (cdf.total_weight > 1e-6f) ? (1.0f / cdf.total_weight) : 0.0f;
        for (int v = 0; v < GRID_HALF_RES; v++) {
            running += cdf.row_sums[v];
            cdf.marginal_cdf[v] = running * inv_total;
        }
        if (GRID_HALF_RES > 0) cdf.marginal_cdf[GRID_HALF_RES - 1] = 1.0f;
        
        for (int v = 0; v < GRID_HALF_RES; v++) {
            const int row_offset = v * GRID_RES;
            float row_sum = cdf.row_sums[v];
            
            if (row_sum < 1e-6f) {
                for (int u = 0; u < GRID_RES; u++) {
                    cdf.row_cdfs[row_offset + u] = (u + 1) * GRID_INV_RES;
                }
            } else {
                float running_row = 0.0f;
                float inv_row_sum = 1.0f / row_sum;
                for (int u = 0; u < GRID_RES; u++) {
                    running_row += cdf.pdf[row_offset + u];
                    cdf.row_cdfs[row_offset + u] = running_row * inv_row_sum;
                }
                cdf.row_cdfs[row_offset + GRID_RES - 1] = 1.0f;
            }
        }
        
        for (int v = GRID_HALF_RES; v < GRID_RES; v++) {
            const int row_offset = v * GRID_RES;
            for (int u = 0; u < GRID_RES; u++) {
                cdf.row_cdfs[row_offset + u] = (u + 1) * GRID_INV_RES;
            }
        }
        
        cdf.is_valid = (cdf.total_weight > 1e-6f) ? 1 : 0;
        
        if (p > 0 && p % progress_step == 0) {
            std::cout << "[CDF] Progress: " << (100 * p / num_primitives) << "%" << std::endl;
        }
    }
    
    cudaMemcpy(d_precomputed_cdfs, h_precomputed_cdfs, 
               num_primitives * sizeof(PrecomputedCDF), cudaMemcpyHostToDevice);
    
    h_scene.precomputed_cdfs = d_precomputed_cdfs;
    if (d_scene) {
        cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[CDF] Complete! " << num_primitives << " CDFs built in " << ms << "ms" << std::endl;
}

inline void SceneState::precomputeCDFsFromFiltered() {
    if (num_primitives == 0 || d_filtered_radiosity == nullptr) return;
    
    std::cout << "\n[CDF-Filtered] Pre-computing CDFs from filtered buffer..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    if (!h_precomputed_cdfs) {
        h_precomputed_cdfs = new PrecomputedCDF[num_primitives];
    }
    
    if (!d_precomputed_cdfs) {
        cudaMalloc(&d_precomputed_cdfs, num_primitives * sizeof(PrecomputedCDF));
    }
    
    size_t filter_size = (size_t)num_primitives * GRID_SIZE * sizeof(float);
    float* h_filtered = new float[num_primitives * GRID_SIZE];
    cudaMemcpy(h_filtered, d_filtered_radiosity, filter_size, cudaMemcpyDeviceToHost);
    
    int progress_step = (num_primitives > 100) ? num_primitives / 10 : 100;
    
    #pragma omp parallel for schedule(dynamic) if(num_primitives > 100)
    for (int p = 0; p < num_primitives; p++) {
        PrecomputedCDF& cdf = h_precomputed_cdfs[p];
        const float* filtered_pdf = h_filtered + p * GRID_SIZE;
        
        for (int i = 0; i < GRID_SIZE; i++) {
            cdf.pdf[i] = filtered_pdf[i];
        }
        
        cdf.total_weight = 0.0f;
        for (int v = 0; v < GRID_HALF_RES; v++) {
            float row_sum = 0.0f;
            const int row_offset = v * GRID_RES;
            for (int u = 0; u < GRID_RES; u++) {
                row_sum += cdf.pdf[row_offset + u];
            }
            cdf.row_sums[v] = row_sum;
            cdf.total_weight += row_sum;
        }
        
        float running = 0.0f;
        float inv_total = (cdf.total_weight > 1e-6f) ? (1.0f / cdf.total_weight) : 0.0f;
        for (int v = 0; v < GRID_HALF_RES; v++) {
            running += cdf.row_sums[v];
            cdf.marginal_cdf[v] = running * inv_total;
        }
        if (GRID_HALF_RES > 0) cdf.marginal_cdf[GRID_HALF_RES - 1] = 1.0f;
        
        for (int v = 0; v < GRID_HALF_RES; v++) {
            const int row_offset = v * GRID_RES;
            float row_sum = cdf.row_sums[v];
            
            if (row_sum < 1e-6f) {
                for (int u = 0; u < GRID_RES; u++) {
                    cdf.row_cdfs[row_offset + u] = (u + 1) * GRID_INV_RES;
                }
            } else {
                float running_row = 0.0f;
                float inv_row_sum = 1.0f / row_sum;
                for (int u = 0; u < GRID_RES; u++) {
                    running_row += cdf.pdf[row_offset + u];
                    cdf.row_cdfs[row_offset + u] = running_row * inv_row_sum;
                }
                cdf.row_cdfs[row_offset + GRID_RES - 1] = 1.0f;
            }
        }
        
        for (int v = GRID_HALF_RES; v < GRID_RES; v++) {
            const int row_offset = v * GRID_RES;
            for (int u = 0; u < GRID_RES; u++) {
                cdf.row_cdfs[row_offset + u] = (u + 1) * GRID_INV_RES;
            }
        }
        
        cdf.is_valid = (cdf.total_weight > 1e-6f) ? 1 : 0;
        
        if (p > 0 && p % progress_step == 0) {
            std::cout << "[CDF-Filtered] Progress: " << (100 * p / num_primitives) << "%" << std::endl;
        }
    }
    
    delete[] h_filtered;
    
    cudaMemcpy(d_precomputed_cdfs, h_precomputed_cdfs, 
               num_primitives * sizeof(PrecomputedCDF), cudaMemcpyHostToDevice);
    
    h_scene.precomputed_cdfs = d_precomputed_cdfs;
    h_scene.use_filtered = false;
    if (d_scene) {
        cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[CDF-Filtered] Complete! CDFs built in " << ms << "ms" << std::endl;
}

// ============================================================================
// RADIOSITY STATE IMPLEMENTATION
// ============================================================================

inline void RadiosityState::runSolver(Primitive* h_primitives, int num_prims,
                                      BVHNode* d_bvh, int* d_indices,
                                      bool enable_filtering, bool use_bilateral, 
                                      float filter_sigma_spatial, float filter_sigma_range) {
    std::cout << "\n========== RADIOSITY SOLVER (OPTIMIZED) ==========\n"
              << "Method: " << (use_monte_carlo ? "Monte Carlo" : "Point-to-Point") << std::endl;
    if (use_monte_carlo) std::cout << "MC Samples: " << mc_samples << std::endl;
    std::cout << "Primitives: " << num_prims << " (Total pairs: " << (num_prims * num_prims) << ")" << std::endl;
    
    for (int i = 0; i < num_prims; ++i) {
        Vector3f Le = h_primitives[i].getLe();
        h_primitives[i].setRadiosity(Le);
        h_primitives[i].setUnshotRad(Le);
    }
    
    cleanup();
    cudaMallocSafe(&d_radiosity_primitives, num_prims * sizeof(Primitive), "d_radiosity_primitives");
    cudaMallocSafe(&d_form_factors, (size_t)num_prims * num_prims * sizeof(float), "d_form_factors");
    cudaMemcpy(d_radiosity_primitives, h_primitives, num_prims * sizeof(Primitive), cudaMemcpyHostToDevice);
    
    // Initialize grids with larger block size for better occupancy
    int block_size = 256;
    int grid_size = (num_prims + block_size - 1) / block_size;
    initialize_directional_grids<<<grid_size, block_size>>>(d_radiosity_primitives, num_prims);
    cudaDeviceSynchronize();
    
    size_t num_pairs = (size_t)num_prims * num_prims;
    cudaMallocSafe(&d_rand_states, num_pairs * sizeof(curandState), "d_radiosity_rand_states");
    
    std::cout << "Initializing random states..." << std::endl;
    auto rand_start = std::chrono::high_resolution_clock::now();
    int init_threads = 256, init_blocks = (num_pairs + init_threads - 1) / init_threads;
    formfactor_rand_init<<<init_blocks, init_threads>>>(num_pairs, d_rand_states);
    cudaDeviceSynchronize();
    auto rand_end = std::chrono::high_resolution_clock::now();
    std::cout << "Random states initialized in " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(rand_end - rand_start).count() << " ms" << std::endl;
    
    std::cout << "Calculating form factors..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Use 8x8 blocks for better occupancy on modern GPUs
    dim3 threads(8, 8), blocks((num_prims + 7) / 8, (num_prims + 7) / 8);
    
    if (use_monte_carlo) {
        calculate_form_factors_mc_kernel<<<blocks, threads>>>(
            d_form_factors, d_radiosity_primitives, num_prims, mc_samples, d_rand_states, d_bvh, d_indices);
    } else {
        calculate_form_factors_kernel<<<blocks, threads>>>(
            d_form_factors, d_radiosity_primitives, num_prims, d_bvh, d_indices);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Form factors calculated in " << duration.count() << " ms" << std::endl;
    
    std::cout << "Running radiosity iterations..." << std::endl;
    dim3 grid_rad((num_prims + 255) / 256, 1), block_rad(256, 1);
    
    for (int i = 0; i < num_iterations; ++i) {
        store_radiosity_history_kernel<<<grid_rad, block_rad>>>(d_radiosity_primitives, num_prims);
        cudaDeviceSynchronize();

        radiosity_iteration_kernel<<<grid_rad, block_rad>>>(d_radiosity_primitives, d_form_factors, num_prims);
        cudaDeviceSynchronize();
        
        update_radiosity_grid<<<grid_rad, block_rad>>>(d_radiosity_primitives, d_form_factors, num_prims);
        cudaDeviceSynchronize();
        
        if (enable_filtering) {
            if (use_bilateral) {
                filter_radiosity_grids(d_radiosity_primitives, num_prims, GRID_RES,
                    filter_sigma_spatial, filter_sigma_range);
            } else {
                filter_radiosity_grids_gaussian(d_radiosity_primitives, num_prims, GRID_RES,
                    filter_sigma_spatial);
            }
        }
        
        if ((i + 1) % 5 == 0 || i == 0) {
            std::cout << "  Iteration " << i + 1 << " complete" << std::endl;
        }
    }

    cudaMemcpy(h_primitives, d_radiosity_primitives, num_prims * sizeof(Primitive), cudaMemcpyDeviceToHost);
    is_calculated = true;
    
    std::cout << "========== RADIOSITY COMPLETE ==========\n" << std::endl;
}

inline void RadiosityState::cleanup() {
    if (d_radiosity_primitives) cudaFree(d_radiosity_primitives);
    if (d_form_factors) cudaFree(d_form_factors);
    if (d_rand_states) cudaFree(d_rand_states);
    
    d_radiosity_primitives = nullptr;
    d_form_factors = nullptr;
    d_rand_states = nullptr;
}

#endif // APPLICATION_STATE_H
