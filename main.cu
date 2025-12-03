#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog.h"

// Project headers
#include "bvh.h"
#include "vector.h"
#include "ray.h"
#include "sensor.h"
#include "triangle.h"
#include "quad.h"
#include "scene.h"
#include "file_manager.h"
#include "render_config.h"
#include "integrator.h"
#include "surface_interaction_record.h"
#include "form_factors.h"

// ============================================================================
// CONSTANTS & ENUMS
// ============================================================================

// Grid configuration is centralized in render_config.h (GRID_RES, GRID_SIZE)
#define RADIOSITY_HISTORY 10


constexpr int   DEFAULT_WIDTH       = 800;
constexpr int   DEFAULT_HEIGHT      = 800;
constexpr float MOUSE_SENSITIVITY   = 0.25f;
constexpr float ZOOM_SENSITIVITY    = 0.1f;

enum class IntegratorType {
    PathTracing,
    Radiosity
};

enum class GridVisualizationMode { VisibilityCount, RadiosityDistribution };

// Extended sampling mode enum with histogram options
enum class HistogramSamplingSource {
    FORM_FACTOR = 0,    // Sample from form factor grid
    RADIOSITY = 1,      // Sample from radiosity grid
    BSDF_WEIGHTED = 2,  // Blend with BSDF sampling
    MIS = 3             // Multiple Importance Sampling
};

// ============================================================================
// CUDA UTILITIES
// ============================================================================

inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error [" << msg << "]: " << cudaGetErrorString(err) << std::endl;
    }
}

template<typename T>
inline void cudaMallocSafe(T** ptr, size_t size, const char* name) {
    cudaError_t err = cudaMalloc((void**)ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate " << name << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
}

// ============================================================================
// STATE STRUCTURES
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
    
    void allocateBuffers();
    void updateResolution(int w, int h);
    void cleanup();
};

struct SceneState {
    Primitive *h_primitives, *d_primitives;
    BVHNode* d_bvh_nodes;
    int* d_bvh_indices;
    Scene* d_scene;
    int num_primitives;
    std::string scene_file;
    
    SceneState() : h_primitives(nullptr), d_primitives(nullptr),
                   d_bvh_nodes(nullptr), d_bvh_indices(nullptr),
                   d_scene(nullptr), num_primitives(0),
                   scene_file("./scenes/cbox_quads.obj") {}
    
    void loadScene(const std::string& filename, int subdivision_count = 0);
    void cleanup();
};

struct RadiosityState {
    Primitive* d_radiosity_primitives;
    float* d_form_factors;
    curandState* d_rand_states;
    int num_iterations, mc_samples;
    bool use_monte_carlo, is_calculated;
    
    RadiosityState() : d_radiosity_primitives(nullptr), d_form_factors(nullptr),
                       d_rand_states(nullptr), num_iterations(10), mc_samples(64),
                       use_monte_carlo(true), is_calculated(false) {}
    
    void runSolver(Primitive* h_prims, int num_prims, BVHNode* d_bvh, int* d_indices);
    void cleanup();
};

struct UIState {
    bool is_dragging, show_grid_window;
    double last_mouse_x, last_mouse_y;
    int hovered_primitive_idx;
    float* h_grid_data;
    GridVisualizationMode grid_viz_mode;
    
    // Histogram configuration
    int grid_resolution;         // Current grid resolution (e.g., 50, 100)
    int top_k_count;            // How many top cells to show/use (0 = all)
    HistogramSamplingSource histogram_source;  // Which histogram to sample from
    bool show_top_k_only;       // Visualize only top-K cells
    bool visualize_luminance;   // Show luminance instead of RGB for radiosity
    
    // Top-K data for current primitive
    int* top_k_indices;         // Indices of top-K cells
    float* top_k_values;        // Values of top-K cells
    int actual_top_k_count;     // Actual count (min of top_k_count and available)
    
    UIState() : is_dragging(false), show_grid_window(false),
                last_mouse_x(0.0), last_mouse_y(0.0),
                hovered_primitive_idx(-1), h_grid_data(nullptr),
                grid_viz_mode(GridVisualizationMode::VisibilityCount),
                grid_resolution(50), top_k_count(0),
                histogram_source(HistogramSamplingSource::FORM_FACTOR),
                show_top_k_only(false), visualize_luminance(true),  // Default to luminance (matches sampling)
                top_k_indices(nullptr), top_k_values(nullptr), actual_top_k_count(0) {}
    
    void cleanup();
};

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
    int current_grid_res;  // NEW: Runtime grid resolution
    
    // Grid filtering options (see grid_filter.h)
    bool enable_grid_filtering;      // Enable/disable bilateral filtering
    bool use_bilateral_filter;       // true = bilateral, false = gaussian
    float filter_sigma_spatial;      // Spatial sigma for filtering
    float filter_sigma_range;        // Range sigma for bilateral filtering
    
    AppConfig() : spp(1), subdivision_step(0), radiosity_step(5),
                  history_step1(0), history_step2(1),
                  current_integrator(IntegratorType::PathTracing),
                  camera_origin(0.5, 3, 8.5), look_at(0, 2.5, 0),
                  up(0, 1, 0), fov(40.0f), sampling_mode(SamplingMode::SAMPLING_BSDF),
                  histogram_source(HistogramSamplingSource::FORM_FACTOR),
                  convert_quads_to_triangles(false), top_k_samples(0),
                  current_grid_res(DEFAULT_GRID_RES),
                  enable_grid_filtering(false), use_bilateral_filter(true),
                  filter_sigma_spatial(1.5f), filter_sigma_range(0.3f) {}
};



struct ApplicationState {
    RenderState render;
    SceneState scene;
    RadiosityState radiosity;
    UIState ui;
    AppConfig config;
} g_state;

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

void renderControlsWindow();
void renderGridWindow();
void renderFrame();
void renderOpenGL();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

__global__ void pick_primitive_kernel(
    int mouse_x, int mouse_y, int width, int height,
    int* result_idx, float* result_dist,
    Primitive* primitives, int num_prims,
    Vector3f cam_origin, Vector3f cam_lower_left,
    Vector3f cam_horizontal, Vector3f cam_vertical);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void pickPrimitive(int mouse_x, int mouse_y, int width, int height,
                   Primitive* d_prims, int num_prims, Sensor& camera, int& prim_idx) {
    if (!d_prims || num_prims == 0) {
        prim_idx = -1;
        return;
    }
    
    int* d_result;
    float* d_dist;
    cudaMalloc(&d_result, sizeof(int));
    cudaMalloc(&d_dist, sizeof(float));
    
    int init_val = -1;
    float init_dist = 1e10f;
    cudaMemcpy(d_result, &init_val, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, &init_dist, sizeof(float), cudaMemcpyHostToDevice);
    
    pick_primitive_kernel<<<1, 1>>>(
        mouse_x, mouse_y, width, height,
        d_result, d_dist, d_prims, num_prims,
        camera.origin, camera.lower_left_corner,
        camera.horizontal, camera.vertical
    );
    
    cudaDeviceSynchronize();
    cudaMemcpy(&prim_idx, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_result);
    cudaFree(d_dist);
}

void copyGridData(Primitive* d_prims, int prim_idx, float* h_grid, 
                  int num_prims, bool use_radiosity = false) {
    if (prim_idx < 0 || prim_idx >= num_prims || !d_prims) {
        memset(h_grid, 0, GRID_SIZE * 3 * sizeof(float));
        return;
    }
    
    // Copy primitive from device to host
    Primitive temp_prim;
    cudaMemcpy(&temp_prim, &d_prims[prim_idx], sizeof(Primitive), cudaMemcpyDeviceToHost);
    
    // IMPORTANT: After copying, we access the grid arrays DIRECTLY from the copied struct
    // (the grid data is inline in Triangle/Quad, so it's copied with the Primitive)
    if (use_radiosity) {
        // Access radiosity_grid inline array from the copied struct
        if (temp_prim.type == PRIM_TRIANGLE) {
            for (int i = 0; i < GRID_SIZE; i++) {
                h_grid[i * 3 + 0] = temp_prim.tri.radiosity_grid[i].x();
                h_grid[i * 3 + 1] = temp_prim.tri.radiosity_grid[i].y();
                h_grid[i * 3 + 2] = temp_prim.tri.radiosity_grid[i].z();
            }
        } else {
            for (int i = 0; i < GRID_SIZE; i++) {
                h_grid[i * 3 + 0] = temp_prim.quad.radiosity_grid[i].x();
                h_grid[i * 3 + 1] = temp_prim.quad.radiosity_grid[i].y();
                h_grid[i * 3 + 2] = temp_prim.quad.radiosity_grid[i].z();
            }
        }
    } else {
        // Access grid inline array from the copied struct
        if (temp_prim.type == PRIM_TRIANGLE) {
            for (int i = 0; i < GRID_SIZE; i++) {
                h_grid[i * 3 + 0] = h_grid[i * 3 + 1] = h_grid[i * 3 + 2] = temp_prim.tri.grid[i];
            }
        } else {
            for (int i = 0; i < GRID_SIZE; i++) {
                h_grid[i * 3 + 0] = h_grid[i * 3 + 1] = h_grid[i * 3 + 2] = temp_prim.quad.grid[i];
            }
        }
    }
}

// ============================================================================
// CUDA KERNELS
// ============================================================================

__global__ void pick_primitive_kernel(
    int mouse_x, int mouse_y, int width, int height,
    int* result_idx, float* result_dist,
    Primitive* primitives, int num_prims,
    Vector3f cam_origin, Vector3f cam_lower_left,
    Vector3f cam_horizontal, Vector3f cam_vertical) 
{
    float u = (float)mouse_x / (float)width;
    float v = 1.0f - ((float)mouse_y / (float)height);
    
    Vector3f direction = unit_vector(cam_lower_left + u * cam_horizontal + 
                                     v * cam_vertical - cam_origin);
    Ray r(cam_origin, direction);
    float closest_t = 1e10f;
    int closest_idx = -1;
    SurfaceInteractionRecord si;
    
    for (int i = 0; i < num_prims; i++) {
        if (primitives[i].intersect(r, 0.001f, closest_t, si)) {
            closest_t = si.t;
            closest_idx = i;
        }
    }
    
    *result_idx = closest_idx;
    *result_dist = closest_t;
}

// ============================================================================
// RENDER STATE IMPLEMENTATION
// ============================================================================

void RenderState::allocateBuffers() {
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

void RenderState::updateResolution(int w, int h) {
    width = w;
    height = h;
    allocateBuffers();
}

void RenderState::cleanup() {
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

// Add this function right before the SceneState::loadScene() implementation
std::vector<Primitive> convertQuadsToTriangles(const std::vector<Primitive>& primitives) {
    std::vector<Primitive> triangulated;
    triangulated.reserve(primitives.size() * 2);
    
    int quad_count = 0, tri_count = 0;
    
    for (const auto& prim : primitives) {
        if (prim.type == PRIM_QUAD) {
            quad_count++;
            const Quad& q = prim.quad;
            
            // Triangle 1: (v00, v10, v11)
            Primitive tri1_prim;
            tri1_prim.type = PRIM_TRIANGLE;
            new (&tri1_prim.tri) Triangle(q.v00, q.v10, q.v11, q.bsdf);
            tri1_prim.tri.Le = q.Le;
            tri1_prim.tri.radiosity = q.radiosity;
            tri1_prim.tri.unshot_rad = q.unshot_rad;
            memcpy(tri1_prim.tri.grid, q.grid, sizeof(q.grid));
            memcpy(tri1_prim.tri.radiosity_grid, q.radiosity_grid, sizeof(q.radiosity_grid));
            triangulated.push_back(tri1_prim);
            
            // Triangle 2: (v00, v11, v01)
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


// ============================================================================
// SCENE STATE IMPLEMENTATION
// ============================================================================

void SceneState::loadScene(const std::string& filename, int subdivision_count) {
    std::cout << "\n========== LOADING SCENE ==========\n"
              << "File: " << filename << "\n"
              << "Subdivision: " << subdivision_count << std::endl;
    
    cleanup();
    
    // Load OBJ file
    Primitive* temp_primitives = nullptr;
    int temp_num_primitives = 0;
    
    if (!loadOBJ(filename, &temp_primitives, temp_num_primitives)) {
        std::cerr << "ERROR: Failed to load scene: " << filename << std::endl;
        return;
    }
    
    std::cout << "Loaded " << temp_num_primitives << " primitives" << std::endl;
    
    // Apply subdivision if requested
    std::vector<Primitive> primitives(temp_primitives, temp_primitives + temp_num_primitives);
    delete[] temp_primitives;

    if (g_state.config.convert_quads_to_triangles) {
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
    
    // Copy to device
    cudaMallocSafe(&d_primitives, num_primitives * sizeof(Primitive), "d_primitives");
    cudaMemcpy(d_primitives, h_primitives, num_primitives * sizeof(Primitive), cudaMemcpyHostToDevice);
    
    // Build BVH
    std::cout << "Building BVH..." << std::endl;
    auto bvh_start = std::chrono::high_resolution_clock::now();
    BVHBuilder builder(h_primitives, num_primitives);
    auto bvh_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(bvh_end - bvh_start);
    std::cout << "BVH built in " << duration.count() << " ms" << std::endl;
    
    // Copy BVH to device
    size_t nodes_size = builder.nodes.size() * sizeof(BVHNode);
    size_t indices_size = builder.primitive_indices.size() * sizeof(int);
    
    cudaMallocSafe(&d_bvh_nodes, nodes_size, "d_bvh_nodes");
    cudaMallocSafe(&d_bvh_indices, indices_size, "d_bvh_indices");
    cudaMemcpy(d_bvh_nodes, builder.nodes.data(), nodes_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bvh_indices, builder.primitive_indices.data(), indices_size, cudaMemcpyHostToDevice);
    
    // Create scene
    Scene h_scene(d_primitives, num_primitives, d_bvh_nodes, d_bvh_indices);
    cudaMallocSafe(&d_scene, sizeof(Scene), "d_scene");
    cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);
    
    scene_file = filename;
    
    std::cout << "========== SCENE LOADED SUCCESSFULLY ==========\n"
              << "Total primitives: " << num_primitives << "\n"
              << "Memory: " << (num_primitives * sizeof(Primitive) + nodes_size + indices_size) / 1024.0 
              << " KB\n" << std::endl;
}

void SceneState::cleanup() {
    if (h_primitives) delete[] h_primitives;
    if (d_primitives) cudaFree(d_primitives);
    if (d_bvh_nodes) cudaFree(d_bvh_nodes);
    if (d_bvh_indices) cudaFree(d_bvh_indices);
    if (d_scene) cudaFree(d_scene);
    
    h_primitives = d_primitives = nullptr;
    d_bvh_nodes = nullptr;
    d_bvh_indices = nullptr;
    d_scene = nullptr;
    num_primitives = 0;
}

// ============================================================================
// RADIOSITY STATE IMPLEMENTATION
// ============================================================================

void RadiosityState::runSolver(Primitive* h_primitives, int num_prims,
                                BVHNode* d_bvh, int* d_indices) {
    std::cout << "\n========== RADIOSITY SOLVER ==========\n"
              << "Method: " << (use_monte_carlo ? "Monte Carlo" : "Point-to-Point") << std::endl;
    if (use_monte_carlo) std::cout << "MC Samples: " << mc_samples << std::endl;
    
    // Initialize radiosity values
    for (int i = 0; i < num_prims; ++i) {
        Vector3f Le = h_primitives[i].getLe();
        h_primitives[i].setRadiosity(Le);
        h_primitives[i].setUnshotRad(Le);
    }
    
    // Allocate GPU memory
    cleanup();
    cudaMallocSafe(&d_radiosity_primitives, num_prims * sizeof(Primitive), "d_radiosity_primitives");
    cudaMallocSafe(&d_form_factors, (size_t)num_prims * num_prims * sizeof(float), "d_form_factors");
    cudaMemcpy(d_radiosity_primitives, h_primitives, num_prims * sizeof(Primitive), cudaMemcpyHostToDevice);
    
    // Initialize directional grids
    dim3 grid_init((num_prims + 255) / 256, 1), block_init(256, 1);
    initialize_directional_grids<<<grid_init, block_init>>>(d_radiosity_primitives, num_prims);
    cudaDeviceSynchronize();
    
    // Initialize random states
    size_t num_pairs = (size_t)num_prims * num_prims;
    cudaMallocSafe(&d_rand_states, num_pairs * sizeof(curandState), "d_radiosity_rand_states");
    int init_threads = 256, init_blocks = (num_pairs + init_threads - 1) / init_threads;
    formfactor_rand_init<<<init_blocks, init_threads>>>(num_pairs, d_rand_states);
    cudaDeviceSynchronize();
    
    // Calculate form factors
    std::cout << "Calculating form factors..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    dim3 threads(16, 16), blocks((num_prims + 15) / 16, (num_prims + 15) / 16);
    
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
    
    // Run iterative solver
    std::cout << "Running radiosity iterations..." << std::endl;
    dim3 grid_rad((num_prims + 255) / 256, 1), block_rad(256, 1);
    
    for (int i = 0; i < num_iterations; ++i) {
        // Store the CURRENT state BEFORE updating
        store_radiosity_history_kernel<<<grid_rad, block_rad>>>(
            d_radiosity_primitives, num_prims);
        cudaDeviceSynchronize();

        // Update radiosity
        radiosity_iteration_kernel<<<grid_rad, block_rad>>>(
            d_radiosity_primitives, d_form_factors, num_prims);
        cudaDeviceSynchronize();
        
        // Update grid
        update_radiosity_grid<<<grid_rad, block_rad>>>(
            d_radiosity_primitives, d_form_factors, num_prims);
        cudaDeviceSynchronize();
        
        // Apply grid filtering if enabled (see grid_filter.h)
        if (g_state.config.enable_grid_filtering) {
            if (g_state.config.use_bilateral_filter) {
                filter_radiosity_grids(d_radiosity_primitives, num_prims, GRID_RES,
                    g_state.config.filter_sigma_spatial, g_state.config.filter_sigma_range);
            } else {
                filter_radiosity_grids_gaussian(d_radiosity_primitives, num_prims, GRID_RES,
                    g_state.config.filter_sigma_spatial);
            }
        }
        
        if ((i + 1) % 5 == 0 || i == 0) {
            std::cout << "  Iteration " << i + 1 << " complete" << std::endl;
        }
    }


    // Copy results back
    cudaMemcpy(h_primitives, d_radiosity_primitives, num_prims * sizeof(Primitive), cudaMemcpyDeviceToHost);
    is_calculated = true;
    
    // Debug: Check radiosity grid values after solver
    float max_rad_grid = 0.0f;
    int non_zero_cells = 0;
    for (int p = 0; p < std::min(5, num_prims); p++) {
        float prim_max = 0.0f;
        int prim_nonzero = 0;
        if (h_primitives[p].type == PRIM_TRIANGLE) {
            for (int c = 0; c < GRID_SIZE; c++) {
                float intensity = h_primitives[p].tri.radiosity_grid[c].x() +
                                 h_primitives[p].tri.radiosity_grid[c].y() +
                                 h_primitives[p].tri.radiosity_grid[c].z();
                if (intensity > 1e-6f) prim_nonzero++;
                prim_max = fmaxf(prim_max, intensity);
            }
        } else {
            for (int c = 0; c < GRID_SIZE; c++) {
                float intensity = h_primitives[p].quad.radiosity_grid[c].x() +
                                 h_primitives[p].quad.radiosity_grid[c].y() +
                                 h_primitives[p].quad.radiosity_grid[c].z();
                if (intensity > 1e-6f) prim_nonzero++;
                prim_max = fmaxf(prim_max, intensity);
            }
        }
        std::cout << "  Prim " << p << " radiosity grid: max=" << prim_max << ", non-zero=" << prim_nonzero << std::endl;
        max_rad_grid = fmaxf(max_rad_grid, prim_max);
        non_zero_cells += prim_nonzero;
    }
    std::cout << "Overall radiosity grid max: " << max_rad_grid << ", total non-zero: " << non_zero_cells << std::endl;
    
    std::cout << "========== RADIOSITY COMPLETE ==========\\n" << std::endl;
}

void RadiosityState::cleanup() {
    if (d_radiosity_primitives) cudaFree(d_radiosity_primitives);
    if (d_form_factors) cudaFree(d_form_factors);
    if (d_rand_states) cudaFree(d_rand_states);
    
    d_radiosity_primitives = nullptr;
    d_form_factors = nullptr;
    d_rand_states = nullptr;
}

// ============================================================================
// UI STATE IMPLEMENTATION
// ============================================================================

void UIState::cleanup() {
    if (h_grid_data) delete[] h_grid_data;
    if (top_k_indices) delete[] top_k_indices;
    if (top_k_values) delete[] top_k_values;
    
    h_grid_data = nullptr;
    top_k_indices = nullptr;
    top_k_values = nullptr;
    actual_top_k_count = 0;
}

// ============================================================================
// CALLBACK IMPLEMENTATIONS
// ============================================================================

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    if (width > 0 && height > 0) {
        g_state.render.updateResolution(width, height);
        glViewport(0, 0, width, height);
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && !ImGui::GetIO().WantCaptureMouse) {
        g_state.ui.is_dragging = (action == GLFW_PRESS);
        if (g_state.ui.is_dragging) {
            glfwGetCursorPos(window, &g_state.ui.last_mouse_x, &g_state.ui.last_mouse_y);
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (g_state.ui.is_dragging) {
        g_state.render.h_camera.yaw += (xpos - g_state.ui.last_mouse_x) * MOUSE_SENSITIVITY;
        g_state.render.h_camera.pitch += (ypos - g_state.ui.last_mouse_y) * MOUSE_SENSITIVITY;
        g_state.render.h_camera.pitch = fmaxf(fminf(g_state.render.h_camera.pitch, 89.0f), -89.0f);
        g_state.ui.last_mouse_x = xpos;
        g_state.ui.last_mouse_y = ypos;
    }
    
    if (!ImGui::GetIO().WantCaptureMouse && g_state.ui.show_grid_window && g_state.scene.d_primitives) {
        pickPrimitive((int)xpos, (int)ypos, g_state.render.width, g_state.render.height,
                     g_state.scene.d_primitives, g_state.scene.num_primitives,
                     g_state.render.h_camera, g_state.ui.hovered_primitive_idx);
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (!ImGui::GetIO().WantCaptureMouse) {
        g_state.render.h_camera.radius -= yoffset * ZOOM_SENSITIVITY;
    }
}

// ============================================================================
// UI RENDERING FUNCTIONS
// ============================================================================

void renderControlsWindow() {
    ImGui::Begin("Controls");
    
    bool res_changed = false;
    res_changed |= ImGui::SliderInt("Width", &g_state.render.width, 200, 2000);
    res_changed |= ImGui::SliderInt("Height", &g_state.render.height, 200, 2000);
    
    if (res_changed) {
        g_state.render.allocateBuffers();
    }
    
    ImGui::SliderInt("SPP", &g_state.config.spp, 1, 1000);
    
    // Scene loading
    if (ImGui::Button("Browse OBJ")) {
        IGFD::FileDialogConfig config;
        config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".obj", config);
    }
    
    if (ImGui::Checkbox("Convert Quads to Triangles", &g_state.config.convert_quads_to_triangles)) {
        g_state.scene.loadScene(g_state.scene.scene_file, g_state.config.subdivision_step);
        g_state.radiosity.runSolver(g_state.scene.h_primitives, g_state.scene.num_primitives,
                                    g_state.scene.d_bvh_nodes, g_state.scene.d_bvh_indices);
        cudaMemcpy(g_state.scene.d_primitives, g_state.scene.h_primitives,
                  g_state.scene.num_primitives * sizeof(Primitive), cudaMemcpyHostToDevice);
    }

    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            g_state.scene.scene_file = ImGuiFileDialog::Instance()->GetFilePathName();
            g_state.scene.loadScene(g_state.scene.scene_file, g_state.config.subdivision_step);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    const char* sampling_items[] = { "BSDF Sampling", "Grid Sampling", "MIS (Mixed)" };
    static int current_sampling = 1; // Default to Grid
    if (ImGui::Combo("Sampling Mode", &current_sampling, sampling_items, 3)) {
        g_state.config.sampling_mode = (SamplingMode)current_sampling;
    }

    // Radiosity history/delta visualization removed for minimal build
    
    // Integrator selection (minimal: PathTracing or Radiosity)
    const char* integrator_names[] = { "Path Tracing", "Radiosity" };
    int current_integrator = (int)g_state.config.current_integrator;
    if (ImGui::Combo("Integrator", &current_integrator, integrator_names, 2)) {
        g_state.config.current_integrator = (IntegratorType)current_integrator;
    }

    
    // Radiosity controls
    bool rad_changed = ImGui::SliderInt("Radiosity Steps", &g_state.radiosity.num_iterations, 0, 50);
    ImGui::Checkbox("Use Monte Carlo", &g_state.radiosity.use_monte_carlo);
    if (g_state.radiosity.use_monte_carlo) {
        ImGui::SliderInt("MC Samples", &g_state.radiosity.mc_samples, 4, 256);
    }
    
    // Grid Filtering Controls (see grid_filter.h)
    ImGui::Separator();
    ImGui::Text("Grid Filtering:");
    ImGui::Checkbox("Enable Grid Filtering (during radiosity)", &g_state.config.enable_grid_filtering);
    if (g_state.config.enable_grid_filtering) {
        ImGui::Checkbox("Use Bilateral (vs Gaussian)", &g_state.config.use_bilateral_filter);
        ImGui::SliderFloat("Spatial Sigma", &g_state.config.filter_sigma_spatial, 0.5f, 5.0f);
        if (g_state.config.use_bilateral_filter) {
            ImGui::SliderFloat("Range Sigma", &g_state.config.filter_sigma_range, 0.05f, 1.0f);
        }
    }
    
    // Apply filter now button (applies to currently loaded grids)
    if (ImGui::Button("Apply Filter Now")) {
        if (g_state.config.use_bilateral_filter) {
            filter_radiosity_grids(g_state.scene.d_primitives, g_state.scene.num_primitives, GRID_RES,
                g_state.config.filter_sigma_spatial, g_state.config.filter_sigma_range);
        } else {
            filter_radiosity_grids_gaussian(g_state.scene.d_primitives, g_state.scene.num_primitives, GRID_RES,
                g_state.config.filter_sigma_spatial);
        }
        // Sync back to host for consistency
        cudaMemcpy(g_state.scene.h_primitives, g_state.scene.d_primitives,
                  g_state.scene.num_primitives * sizeof(Primitive), cudaMemcpyDeviceToHost);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Apply bilateral/gaussian filtering to the current radiosity grids.\\n"
                          "This smooths noise while preserving important directional features.");
    }
    ImGui::Separator();
    
    if (ImGui::Button("Calculate Radiosity") || rad_changed) {
        g_state.radiosity.runSolver(g_state.scene.h_primitives, g_state.scene.num_primitives,
                                    g_state.scene.d_bvh_nodes, g_state.scene.d_bvh_indices);
        cudaMemcpy(g_state.scene.d_primitives, g_state.scene.h_primitives,
                  g_state.scene.num_primitives * sizeof(Primitive), cudaMemcpyHostToDevice);
    }
    
    // Save image
    if (ImGui::Button("Save PNG")) {
        IGFD::FileDialogConfig config;
        config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog("SaveFileDlgKey", "Save PNG", ".png", config);
    }
    
    if (ImGuiFileDialog::Instance()->Display("SaveFileDlgKey")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string path = ImGuiFileDialog::Instance()->GetFilePathName();
            if (path.substr(path.size() - 4) != ".png") path += ".png";
            stbi_flip_vertically_on_write(1);
            stbi_write_png(path.c_str(), g_state.render.width, g_state.render.height, 3,
                          g_state.render.h_image, g_state.render.width * 3);
        }
        ImGuiFileDialog::Instance()->Close();
    }
    
    // Subdivision
    bool subdiv_changed = ImGui::SliderInt("Subdivision", &g_state.config.subdivision_step, 0, 10);
    if (subdiv_changed) {
        g_state.scene.loadScene(g_state.scene.scene_file, g_state.config.subdivision_step);
        g_state.radiosity.runSolver(g_state.scene.h_primitives, g_state.scene.num_primitives,
                                    g_state.scene.d_bvh_nodes, g_state.scene.d_bvh_indices);
        cudaMemcpy(g_state.scene.d_primitives, g_state.scene.h_primitives,
                  g_state.scene.num_primitives * sizeof(Primitive), cudaMemcpyHostToDevice);
    }
    
    // Grid visualization
    ImGui::Checkbox("Show Grid Window", &g_state.ui.show_grid_window);
    
    ImGui::Separator();
    ImGui::Text("Scene Statistics:");
    ImGui::Text("  Total Primitives: %d", g_state.scene.num_primitives);
    
    if (g_state.scene.h_primitives) {
        int tri_count = 0, quad_count = 0;
        for (int i = 0; i < g_state.scene.num_primitives; i++) {
            if (g_state.scene.h_primitives[i].type == PRIM_TRIANGLE) {
                tri_count++;
            } else {
                quad_count++;
            }
        }
        ImGui::Text("  Triangles: %d", tri_count);
        ImGui::Text("  Quads: %d", quad_count);
    }
    
    ImGui::End();
}

void renderGridWindow() {
    if (!g_state.ui.show_grid_window) return;
    
    ImGui::Begin("Directional Grid", &g_state.ui.show_grid_window);

    // ===== SAMPLING MODE INFO =====
    const char* mode_names[] = { "BSDF (cosine)", "Grid (FormFactor)", "MIS (Radiosity+BSDF)" };
    int mode_idx = (g_state.config.sampling_mode == SamplingMode::SAMPLING_BSDF) ? 0 :
                   (g_state.config.sampling_mode == SamplingMode::SAMPLING_FORMFACTOR || 
                    g_state.config.sampling_mode == SamplingMode::SAMPLING_TOPK) ? 1 : 2;
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Current Sampling: %s", mode_names[mode_idx]);
    
    // ===== GRID VISUALIZATION MODE =====
    ImGui::Separator();
    const char* modes[] = { "Form Factor (Grid/TopK sampling)", "Radiosity Luminance (Radiosity/MIS sampling)" };
    int current_mode = (g_state.ui.histogram_source == HistogramSamplingSource::RADIOSITY) ? 1 : 0;
    if (ImGui::Combo("Visualize##viz", &current_mode, modes, 2)) {
        g_state.ui.histogram_source = (current_mode == 0) ? 
            HistogramSamplingSource::FORM_FACTOR : HistogramSamplingSource::RADIOSITY;
        g_state.config.histogram_source = g_state.ui.histogram_source;
        // Auto-enable luminance mode when switching to radiosity (matches what sampler uses)
        if (current_mode == 1) {
            g_state.ui.visualize_luminance = true;
        }
    }
    
    // Show RGB option only for radiosity (for debugging/visualization purposes)
    if (g_state.ui.histogram_source == HistogramSamplingSource::RADIOSITY) {
        bool show_rgb = !g_state.ui.visualize_luminance;
        if (ImGui::Checkbox("Show RGB (debug only, not sampling PDF)", &show_rgb)) {
            g_state.ui.visualize_luminance = !show_rgb;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Sampling uses LUMINANCE: Y = 0.2126R + 0.7152G + 0.0722B\n"
                              "Check this to see RGB colors (doesn't match what sampler sees).");
        }
    }
    
    // ===== GRID RESOLUTION INFO =====
    ImGui::Separator();
    ImGui::Text("Grid Configuration:");
    ImGui::Text("Grid Resolution: %d x %d (fixed at compile time)", GRID_RES, GRID_RES);
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Total cells: %d", GRID_SIZE);
    
    // ===== TOP-K SAMPLING CONTROLS =====
    ImGui::Separator();
    ImGui::Text("Top-K Sampling:");
    
    const char* topk_modes[] = { "Form Factor", "Radiosity Intensity" };
    int current_topk_mode = (g_state.ui.histogram_source == HistogramSamplingSource::FORM_FACTOR) ? 0 : 1;
    if (ImGui::Combo("Top-K Source##main", &current_topk_mode, topk_modes, 2)) {
        g_state.ui.histogram_source = (current_topk_mode == 0) ? 
            HistogramSamplingSource::FORM_FACTOR : HistogramSamplingSource::RADIOSITY;
        g_state.config.histogram_source = g_state.ui.histogram_source;
    }
    
    ImGui::Checkbox("Enable Top-K Sampling##grid", &g_state.ui.show_top_k_only);
    ImGui::SliderInt("Top-K Count##grid", &g_state.ui.top_k_count, 1, 500);
    g_state.config.top_k_samples = g_state.ui.top_k_count;
    
    if (g_state.ui.hovered_primitive_idx >= 0 && 
        g_state.ui.hovered_primitive_idx < g_state.scene.num_primitives) {
        
        ImGui::Separator();
        ImGui::Text("Primitive ID: %d", g_state.ui.hovered_primitive_idx);
        
        // Allocate grid data if needed
        if (!g_state.ui.h_grid_data) {
            g_state.ui.h_grid_data = new float[GRID_RES * GRID_RES * 3];
        }
        
        // Allocate top-K data if needed
        if (!g_state.ui.top_k_indices && g_state.ui.top_k_count > 0) {
            g_state.ui.top_k_indices = new int[g_state.ui.top_k_count];
            g_state.ui.top_k_values = new float[g_state.ui.top_k_count];
        }
        
        // Use histogram_source to determine what to visualize
        bool use_radiosity_grid = (g_state.ui.histogram_source == HistogramSamplingSource::RADIOSITY);
        copyGridData(g_state.scene.d_primitives, g_state.ui.hovered_primitive_idx,
                    g_state.ui.h_grid_data, g_state.scene.num_primitives, use_radiosity_grid);
        
        // Calculate top-K from the selected grid
        if (g_state.ui.show_top_k_only && g_state.ui.top_k_count > 0) {
            Primitive temp_prim;
            cudaMemcpy(&temp_prim, &g_state.scene.d_primitives[g_state.ui.hovered_primitive_idx], 
                      sizeof(Primitive), cudaMemcpyDeviceToHost);
            
            // For radiosity, we need to compute intensity values first
            if (use_radiosity_grid) {
                // Convert RGB radiosity to intensity for top-K calculation
                std::vector<std::pair<int, float>> intensity_pairs;
                intensity_pairs.reserve(GRID_SIZE);
                
                if (temp_prim.type == PRIM_TRIANGLE) {
                    for (int i = 0; i < GRID_SIZE; i++) {
                        float intensity = temp_prim.tri.radiosity_grid[i].x() + 
                                         temp_prim.tri.radiosity_grid[i].y() + 
                                         temp_prim.tri.radiosity_grid[i].z();
                        intensity_pairs.push_back({i, intensity});
                    }
                } else {
                    for (int i = 0; i < GRID_SIZE; i++) {
                        float intensity = temp_prim.quad.radiosity_grid[i].x() + 
                                         temp_prim.quad.radiosity_grid[i].y() + 
                                         temp_prim.quad.radiosity_grid[i].z();
                        intensity_pairs.push_back({i, intensity});
                    }
                }
                
                // Sort and extract top-K
                std::partial_sort(intensity_pairs.begin(), 
                                 intensity_pairs.begin() + std::min(g_state.ui.top_k_count, (int)intensity_pairs.size()),
                                 intensity_pairs.end(),
                                 [](const auto& a, const auto& b) { return a.second > b.second; });
                
                g_state.ui.actual_top_k_count = std::min(g_state.ui.top_k_count, (int)intensity_pairs.size());
                for (int i = 0; i < g_state.ui.actual_top_k_count; i++) {
                    g_state.ui.top_k_indices[i] = intensity_pairs[i].first;
                    g_state.ui.top_k_values[i] = intensity_pairs[i].second;
                }
            } else {
                // Use form-factor grid directly
                temp_prim.getTopKIndices(g_state.ui.top_k_count, g_state.ui.top_k_indices, 
                                         g_state.ui.top_k_values, g_state.ui.actual_top_k_count);
            }
            
            ImGui::Text("Actual Top-K: %d", g_state.ui.actual_top_k_count);
            ImGui::Text("Top-K Max Value: %.3f", g_state.ui.actual_top_k_count > 0 ? 
                                                  g_state.ui.top_k_values[0] : 0.0f);
        }
        
        // Render heatmap
        ImVec2 size(400, 400);
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        
        // Determine if we should use luminance for visualization (matches sampling PDF)
        bool show_luminance = use_radiosity_grid && g_state.ui.visualize_luminance;
        
        // Find max value for normalization
        float max_val = 0.0f;
        if (g_state.ui.show_top_k_only && g_state.ui.actual_top_k_count > 0) {
            max_val = g_state.ui.top_k_values[0];  // First element is max (sorted)
        } else {
            if (use_radiosity_grid) {
                for (int i = 0; i < GRID_RES * GRID_RES; i++) {
                    if (show_luminance) {
                        // Luminance: Y = 0.2126*R + 0.7152*G + 0.0722*B (ITU-R BT.709)
                        float lum = 0.2126f * g_state.ui.h_grid_data[i * 3 + 0] +
                                    0.7152f * g_state.ui.h_grid_data[i * 3 + 1] +
                                    0.0722f * g_state.ui.h_grid_data[i * 3 + 2];
                        max_val = fmaxf(max_val, lum);
                    } else {
                        max_val = fmaxf(max_val, g_state.ui.h_grid_data[i * 3 + 0]);
                        max_val = fmaxf(max_val, g_state.ui.h_grid_data[i * 3 + 1]);
                        max_val = fmaxf(max_val, g_state.ui.h_grid_data[i * 3 + 2]);
                    }
                }
            } else {
                for (int i = 0; i < GRID_RES * GRID_RES; i++) {
                    max_val = fmaxf(max_val, g_state.ui.h_grid_data[i * 3]);
                }
            }
        }
        
        ImGui::Text("Max Value: %.6f", max_val);
        const char* view_mode = use_radiosity_grid ? 
            (show_luminance ? "Radiosity (Luminance/PDF)" : "Radiosity (RGB)") : "Form Factor Grid";
        ImGui::Text("Viewing: %s", view_mode);
        
        // Debug: show sum of all values to verify data is being read
        float total_sum = 0.0f;
        int non_zero_count = 0;
        for (int i = 0; i < GRID_SIZE; i++) {
            float val;
            if (use_radiosity_grid) {
                if (show_luminance) {
                    val = 0.2126f * g_state.ui.h_grid_data[i * 3 + 0] +
                          0.7152f * g_state.ui.h_grid_data[i * 3 + 1] +
                          0.0722f * g_state.ui.h_grid_data[i * 3 + 2];
                } else {
                    val = g_state.ui.h_grid_data[i * 3 + 0] + g_state.ui.h_grid_data[i * 3 + 1] + g_state.ui.h_grid_data[i * 3 + 2];
                }
            } else {
                val = g_state.ui.h_grid_data[i * 3];
            }
            total_sum += val;
            if (val > 1e-6f) non_zero_count++;
        }
        ImGui::Text("Total Sum: %.4f | Non-zero cells: %d", total_sum, non_zero_count);
        
        float cell_w = size.x / GRID_RES;
        float cell_h = size.y / GRID_RES;
        
        // Create a set of top-K indices for fast lookup (only if needed)
        std::unordered_set<int> top_k_set;
        if (g_state.ui.show_top_k_only && g_state.ui.actual_top_k_count > 0) {
            top_k_set.reserve(g_state.ui.actual_top_k_count);
            for (int i = 0; i < g_state.ui.actual_top_k_count; i++) {
                top_k_set.insert(g_state.ui.top_k_indices[i]);
            }
        }
        
        for (int theta = 0; theta < GRID_RES; theta++) {
            for (int phi = 0; phi < GRID_RES; phi++) {
                int idx = theta * GRID_RES + phi;
                
                // Skip if showing top-K only and this isn't in top-K
                if (g_state.ui.show_top_k_only && top_k_set.find(idx) == top_k_set.end()) {
                    continue;
                }
                
                ImU32 color;
                if (use_radiosity_grid) {
                    float r = g_state.ui.h_grid_data[idx * 3 + 0];
                    float g_val = g_state.ui.h_grid_data[idx * 3 + 1];
                    float b = g_state.ui.h_grid_data[idx * 3 + 2];
                    
                    if (show_luminance) {
                        // Luminance mode: use same formula as sampling
                        float lum = 0.2126f * r + 0.7152f * g_val + 0.0722f * b;
                        float val = (max_val > 1e-6f) ? fminf(lum / max_val, 1.0f) : 0.0f;
                        
                        // Use heatmap colormap for luminance (matches form factor display)
                        float hr, hg, hb;
                        if (val < 0.25f) {
                            hr = val * 4.0f;
                            hg = 0.0f;
                            hb = 0.0f;
                        } else if (val < 0.5f) {
                            hr = 1.0f;
                            hg = (val - 0.25f) * 2.0f;
                            hb = 0.0f;
                        } else if (val < 0.75f) {
                            hr = 1.0f;
                            hg = 0.5f + (val - 0.5f) * 2.0f;
                            hb = 0.0f;
                        } else {
                            hr = 1.0f;
                            hg = 1.0f;
                            hb = (val - 0.75f) * 4.0f;
                        }
                        color = IM_COL32((int)(hr * 255), (int)(hg * 255), (int)(hb * 255), 255);
                    } else {
                        // RGB mode: display actual radiosity colors
                        if (max_val > 1e-6f) {
                            r = fminf(r / max_val, 1.0f);
                            g_val = fminf(g_val / max_val, 1.0f);
                            b = fminf(b / max_val, 1.0f);
                        }
                        
                        r = sqrtf(r);
                        g_val = sqrtf(g_val);
                        b = sqrtf(b);
                        
                        color = IM_COL32((int)(r * 255), (int)(g_val * 255), (int)(b * 255), 255);
                    }
                } else {
                    float val = g_state.ui.h_grid_data[idx * 3];
                    if (max_val > 1e-6f) {
                        val = fminf(val / max_val, 1.0f);
                    }
                    
                    // Hot colormap (optimized)
                    float r, g_val, b;
                    if (val < 0.25f) {
                        r = val * 4.0f;
                        g_val = 0.0f;
                        b = 0.0f;
                    } else if (val < 0.5f) {
                        r = 1.0f;
                        g_val = (val - 0.25f) * 2.0f;
                        b = 0.0f;
                    } else if (val < 0.75f) {
                        r = 1.0f;
                        g_val = 0.5f + (val - 0.5f) * 2.0f;
                        b = 0.0f;
                    } else {
                        r = 1.0f;
                        g_val = 1.0f;
                        b = (val - 0.75f) * 4.0f;
                    }
                    
                    color = IM_COL32((int)(r * 255), (int)(g_val * 255), (int)(b * 255), 255);
                }
                
                ImVec2 p_min(canvas_pos.x + phi * cell_w, canvas_pos.y + theta * cell_h);
                ImVec2 p_max(p_min.x + cell_w, p_min.y + cell_h);
                
                draw_list->AddRectFilled(p_min, p_max, color);
            }
        }
        
        // Add a legend/colorbar
        ImGui::Dummy(size);
        
        if (use_radiosity_grid) {
            if (show_luminance) {
                ImGui::Text("Colormap: Luminance heatmap (matches sampling PDF)");
            } else {
                ImGui::Text("Colormap: RGB Radiosity (normalized, gamma corrected)");
            }
        } else {
            ImGui::Text("Colormap: Black -> Red -> Orange -> Yellow -> White");
        }
        
    } else {
        ImGui::Text("Hover over a primitive to view its directional grid");
    }
    
    ImGui::End();
}

void renderFrame() {
    // Update camera
    g_state.render.h_camera.updateCameraOrbit();
    cudaMemcpy(g_state.render.d_camera, &g_state.render.h_camera, sizeof(Sensor), cudaMemcpyHostToDevice);
    
    // Render based on integrator type
    switch (g_state.config.current_integrator) {
        case IntegratorType::PathTracing:
            render<<<g_state.render.grid, g_state.render.block>>>(
                g_state.render.d_image, g_state.render.d_camera, g_state.scene.d_scene,
                g_state.render.d_rand_state, g_state.config.spp, g_state.config.sampling_mode);
            break;

        case IntegratorType::Radiosity:
            render_radiosity<<<g_state.render.grid, g_state.render.block>>>(
                g_state.render.d_image, g_state.render.d_camera, g_state.scene.d_scene,
                g_state.render.d_rand_state, g_state.config.spp);
            break;
    }
    
    cudaDeviceSynchronize();
    
    // Copy to host
    cudaMemcpy(g_state.render.h_image, g_state.render.d_image, g_state.render.img_size,
              cudaMemcpyDeviceToHost);
}

void renderOpenGL() {
    // Update texture
    glBindTexture(GL_TEXTURE_2D, g_state.render.texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_state.render.width, g_state.render.height,
                   GL_RGB, GL_UNSIGNED_BYTE, g_state.render.h_image);
    
    // Clear and setup matrices
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Render textured quad
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_state.render.texture);
    
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();
    
    glDisable(GL_TEXTURE_2D);
}

// ============================================================================
// INITIALIZATION
// ============================================================================

bool initializeOpenGL(GLFWwindow** window) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    
    *window = glfwCreateWindow(DEFAULT_WIDTH, DEFAULT_HEIGHT, "CUDA Path Tracer", NULL, NULL);
    if (!*window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(*window);
    glfwSwapInterval(1);
    glewInit();
    
    // Set callbacks
    glfwSetFramebufferSizeCallback(*window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(*window, mouse_button_callback);
    glfwSetCursorPosCallback(*window, cursor_position_callback);
    glfwSetScrollCallback(*window, scroll_callback);
    
    return true;
}

void initializeImGui(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_ViewportsEnable;
    
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }
    
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
}

bool initializeApplication() {
    // Initialize camera
    g_state.render.h_camera = Sensor(
        g_state.config.camera_origin,
        g_state.config.look_at,
        g_state.config.up,
        g_state.config.fov,
        (float)DEFAULT_WIDTH / (float)DEFAULT_HEIGHT
    );
    
    // Allocate device camera
    cudaMallocSafe(&g_state.render.d_camera, sizeof(Sensor), "d_camera");
    cudaMemcpy(g_state.render.d_camera, &g_state.render.h_camera, sizeof(Sensor), cudaMemcpyHostToDevice);
    
    // Load default scene
    g_state.scene.loadScene(g_state.scene.scene_file, 0);
    
    // Create OpenGL texture
    glGenTextures(1, &g_state.render.texture);
    glBindTexture(GL_TEXTURE_2D, g_state.render.texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // Allocate render buffers
    g_state.render.allocateBuffers();
    
    return true;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv) {
    // Initialize OpenGL and GLFW
    GLFWwindow* window = nullptr;
    if (!initializeOpenGL(&window)) return -1;
    
    // Initialize ImGui
    initializeImGui(window);
    
    // Initialize application state
    if (!initializeApplication()) {
        std::cerr << "Failed to initialize application" << std::endl;
        return -1;
    }
    
    std::cout << "Application initialized successfully\n"
              << "Controls:\n"
              << "  - Left click + drag: Rotate camera\n"
              << "  - Scroll: Zoom in/out\n"
              << "  - Use UI to change settings" << std::endl;
    
    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Start new ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Render UI windows
        renderControlsWindow();
        renderGridWindow();
        
        // Render frame
        renderFrame();
        
        // Render OpenGL quad with texture
        renderOpenGL();
        
        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        // Update platform windows (for multi-viewport)
        ImGuiIO& io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_context);
        }
        
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    std::cout << "\nCleaning up resources..." << std::endl;
    g_state.render.cleanup();
    g_state.scene.cleanup();
    g_state.radiosity.cleanup();
    g_state.ui.cleanup();
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    
    std::cout << "Application terminated successfully" << std::endl;
    return 0;
}
