#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
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
#include "integrator.h"
#include "surface_interaction_record.h"
#include "form_factors.h"

// ============================================================================
// CONSTANTS & ENUMS
// ============================================================================


#define GRID_RES 20
#define GRID_SIZE (GRID_RES * GRID_RES)  // Total grid cells
#define RADIOSITY_HISTORY 10


constexpr int   DEFAULT_WIDTH       = 800;
constexpr int   DEFAULT_HEIGHT      = 800;
constexpr float MOUSE_SENSITIVITY   = 0.25f;
constexpr float ZOOM_SENSITIVITY    = 0.1f;

enum class IntegratorType { 
    PathTracing, 
    Radiosity, 
    RadiosityDelta,
    RadiosityHistory  // New option
};

enum class GridVisualizationMode { VisibilityCount, RadiosityDistribution };

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
    
    UIState() : is_dragging(false), show_grid_window(false),
                last_mouse_x(0.0), last_mouse_y(0.0),
                hovered_primitive_idx(-1), h_grid_data(nullptr),
                grid_viz_mode(GridVisualizationMode::VisibilityCount) {}
    
    void cleanup();
};

struct AppConfig {
    int spp, subdivision_step, radiosity_step;
    int history_step1, history_step2;
    IntegratorType current_integrator;
    Vector3f camera_origin, look_at, up;
    SamplingMode sampling_mode;
    float fov;
    bool convert_quads_to_triangles;
    
    AppConfig() : spp(1), subdivision_step(0), radiosity_step(5),
                  history_step1(0), history_step2(1),  // ADD THESE
                  current_integrator(IntegratorType::PathTracing),
                  camera_origin(0.5, 3, 8.5), look_at(0, 2.5, 0),
                  up(0, 1, 0), fov(40.0f), sampling_mode(SAMPLING_BSDF),
                  convert_quads_to_triangles(false) {}
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
    
    Primitive temp_prim;
    cudaMemcpy(&temp_prim, &d_prims[prim_idx], sizeof(Primitive), cudaMemcpyDeviceToHost);
    
    if (use_radiosity) {
        Vector3f* rad_grid = temp_prim.getRadiosityGrid();
        for (int i = 0; i < GRID_SIZE; i++) {
            h_grid[i * 3 + 0] = rad_grid[i].x();
            h_grid[i * 3 + 1] = rad_grid[i].y();
            h_grid[i * 3 + 2] = rad_grid[i].z();
        }
    } else {
        float* grid = temp_prim.getGrid();
        for (int i = 0; i < GRID_SIZE; i++) {
            h_grid[i * 3 + 0] = h_grid[i * 3 + 1] = h_grid[i * 3 + 2] = grid[i];
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
        
        if ((i + 1) % 5 == 0 || i == 0) {
            std::cout << "  Iteration " << i + 1 << " complete" << std::endl;
        }
    }


    // Copy results back
    cudaMemcpy(h_primitives, d_radiosity_primitives, num_prims * sizeof(Primitive), cudaMemcpyDeviceToHost);
    is_calculated = true;
    std::cout << "========== RADIOSITY COMPLETE ==========\n" << std::endl;
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
    h_grid_data = nullptr;
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

    if (g_state.config.current_integrator == IntegratorType::RadiosityDelta ||
        g_state.config.current_integrator == IntegratorType::RadiosityHistory) {
        ImGui::Text("History Visualization:");
        ImGui::SliderInt("Step 1 (Recent)", &g_state.config.history_step1, 0, RADIOSITY_HISTORY - 1);
        ImGui::SliderInt("Step 2 (Older)", &g_state.config.history_step2, 0, RADIOSITY_HISTORY - 1);
        ImGui::Text("Delta: Step %d - Step %d", g_state.config.history_step1, g_state.config.history_step2);
    }
    
    // Integrator selection
    const char* integrator_names[] = { "Path Tracing", "Radiosity", "Radiosity Delta", "Radiosity History" };
    int current_integrator = (int)g_state.config.current_integrator;
    if (ImGui::Combo("Integrator", &current_integrator, integrator_names, 4)) {  // Changed from 3 to 4
        g_state.config.current_integrator = (IntegratorType)current_integrator;
    }

    
    // Radiosity controls
    bool rad_changed = ImGui::SliderInt("Radiosity Steps", &g_state.radiosity.num_iterations, 0, 50);
    ImGui::Checkbox("Use Monte Carlo", &g_state.radiosity.use_monte_carlo);
    if (g_state.radiosity.use_monte_carlo) {
        ImGui::SliderInt("MC Samples", &g_state.radiosity.mc_samples, 4, 256);
    }
    
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

    const char* modes[] = { "Visibility Count", "Radiosity Distribution" };
    int current_mode = (int)g_state.ui.grid_viz_mode;
    if (ImGui::Combo("Mode", &current_mode, modes, 2)) {
        g_state.ui.grid_viz_mode = (GridVisualizationMode)current_mode;
    }
    
    if (g_state.ui.hovered_primitive_idx >= 0 && 
        g_state.ui.hovered_primitive_idx < g_state.scene.num_primitives) {
        
        ImGui::Text("Primitive ID: %d", g_state.ui.hovered_primitive_idx);
        
        // Allocate grid data if needed
        if (!g_state.ui.h_grid_data) {
            g_state.ui.h_grid_data = new float[GRID_RES * GRID_RES * 3];
        }
        
        bool use_rad = (g_state.ui.grid_viz_mode == GridVisualizationMode::RadiosityDistribution);
        copyGridData(g_state.scene.d_primitives, g_state.ui.hovered_primitive_idx,
                    g_state.ui.h_grid_data, g_state.scene.num_primitives, use_rad);
        
        // Render heatmap
        ImVec2 size(400, 400);
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        
        // Find max value for normalization
        float max_val = 0.0f;
        if (use_rad) {
            // For radiosity, find max across all RGB channels
            for (int i = 0; i < GRID_RES * GRID_RES; i++) {
                max_val = fmaxf(max_val, g_state.ui.h_grid_data[i * 3 + 0]);
                max_val = fmaxf(max_val, g_state.ui.h_grid_data[i * 3 + 1]);
                max_val = fmaxf(max_val, g_state.ui.h_grid_data[i * 3 + 2]);
            }
        } else {
            // For visibility count, find max of grayscale values
            for (int i = 0; i < GRID_RES * GRID_RES; i++) {
                max_val = fmaxf(max_val, g_state.ui.h_grid_data[i * 3]);
            }
        }
        
        ImGui::Text("Max Value: %.3f", max_val);
        
        float cell_w = size.x / GRID_RES;
        float cell_h = size.y / GRID_RES;
        
        for (int theta = 0; theta < GRID_RES; theta++) {
            for (int phi = 0; phi < GRID_RES; phi++) {
                int idx = theta * GRID_RES + phi;
                
                ImU32 color;
                if (use_rad) {
                    // Radiosity: display RGB directly with normalization
                    float r = g_state.ui.h_grid_data[idx * 3 + 0];
                    float g = g_state.ui.h_grid_data[idx * 3 + 1];
                    float b = g_state.ui.h_grid_data[idx * 3 + 2];
                    
                    // Normalize and clamp
                    if (max_val > 1e-6f) {
                        r = fminf(r / max_val, 1.0f);
                        g = fminf(g / max_val, 1.0f);
                        b = fminf(b / max_val, 1.0f);
                    }
                    
                    // Apply gamma correction for better visibility
                    r = powf(r, 0.5f);
                    g = powf(g, 0.5f);
                    b = powf(b, 0.5f);
                    
                    color = IM_COL32((int)(r * 255), (int)(g * 255), (int)(b * 255), 255);
                } else {
                    // Visibility count: use hot colormap (black -> red -> orange -> yellow -> white)
                    float val = g_state.ui.h_grid_data[idx * 3];
                    if (max_val > 1e-6f) {
                        val = fminf(val / max_val, 1.0f);
                    }
                    
                    // Hot colormap
                    float r, g, b;
                    if (val < 0.25f) {
                        // Black to dark red
                        r = val / 0.25f;
                        g = 0.0f;
                        b = 0.0f;
                    } else if (val < 0.5f) {
                        // Dark red to bright red
                        r = 1.0f;
                        g = (val - 0.25f) / 0.25f * 0.5f;
                        b = 0.0f;
                    } else if (val < 0.75f) {
                        // Bright red to orange/yellow
                        r = 1.0f;
                        g = 0.5f + (val - 0.5f) / 0.25f * 0.5f;
                        b = 0.0f;
                    } else {
                        // Yellow to white
                        r = 1.0f;
                        g = 1.0f;
                        b = (val - 0.75f) / 0.25f;
                    }
                    
                    color = IM_COL32((int)(r * 255), (int)(g * 255), (int)(b * 255), 255);
                }
                
                ImVec2 p_min(canvas_pos.x + phi * cell_w, canvas_pos.y + theta * cell_h);
                ImVec2 p_max(p_min.x + cell_w, p_min.y + cell_h);
                
                draw_list->AddRectFilled(p_min, p_max, color);
            }
        }
        
        // Add a legend/colorbar
        ImGui::Dummy(size);
        
        if (!use_rad) {
            ImGui::Text("Colormap: Black -> Red -> Orange -> Yellow -> White");
        } else {
            ImGui::Text("Colormap: RGB Radiosity (normalized)");
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
            
        case IntegratorType::RadiosityDelta:
            if (g_state.radiosity.is_calculated) {
                // Call the delta integrator with history parameters
                radiosity_delta_integrator<<<g_state.render.grid, g_state.render.block>>>(
                    g_state.render.d_image, 
                    g_state.render.width, 
                    g_state.render.height,
                    g_state.render.d_camera, 
                    g_state.scene.d_scene, 
                    g_state.config.history_step1,  // Pass history step 1
                    g_state.config.history_step2); // Pass history step 2
            }
            break;
            
        case IntegratorType::RadiosityHistory:
            // You can implement a different visualization for this mode
            if (g_state.radiosity.is_calculated) {
                radiosity_delta_integrator<<<g_state.render.grid, g_state.render.block>>>(
                    g_state.render.d_image, 
                    g_state.render.width, 
                    g_state.render.height,
                    g_state.render.d_camera, 
                    g_state.scene.d_scene, 
                    g_state.config.history_step1,  // Pass history step 1
                    g_state.config.history_step2); // Pass history step 2
            }
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