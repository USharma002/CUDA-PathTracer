/**
 * @file application.h
 * @brief Application initialization and rendering functions
 * 
 * Contains OpenGL/GLFW initialization, ImGui setup, and main render loop functions.
 */

#ifndef APPLICATION_H
#define APPLICATION_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <iostream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "application_state.h"
#include "ui/callbacks.h"
#include "utils/profiler.h"

// ============================================================================
// OPENGL INITIALIZATION
// ============================================================================

/**
 * @brief Initialize GLFW and create window
 * @param window Output pointer to created window
 * @return true on success, false on failure
 */
inline bool initializeOpenGL(GLFWwindow** window) {
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
    
    // Register callbacks
    registerCallbacks(*window);
    
    return true;
}

// ============================================================================
// IMGUI INITIALIZATION
// ============================================================================

/**
 * @brief Initialize ImGui with GLFW/OpenGL backends
 */
inline void initializeImGui(GLFWwindow* window) {
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

// ============================================================================
// APPLICATION INITIALIZATION
// ============================================================================

/**
 * @brief Initialize all application state
 * @return true on success, false on failure
 */
inline bool initializeApplication() {
    // Initialize profiler stages
    auto& profiler = Profiler::getInstance();
    profiler.addStage("Camera Update", false);
    profiler.addStage("Render Kernel", false);
    profiler.addStage("Memory Transfer", false);
    
    // Allocate kernel profiling buffer
#if ENABLE_KERNEL_PROFILING
    cudaMallocSafe(&d_kernel_profile, sizeof(KernelProfileData), "d_kernel_profile");
    h_kernel_profile.reset();
    cudaMemcpy(d_kernel_profile, &h_kernel_profile, sizeof(KernelProfileData), cudaMemcpyHostToDevice);
#endif
    
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
    g_state.scene.loadScene(g_state.scene.scene_file, 0, g_state.config.convert_quads_to_triangles);
    
    // Run initial radiosity solve and precompute CDFs for fast grid sampling
    // This is CRITICAL for performance - without precomputed CDFs, grid sampling
    // rebuilds CDFs per-sample which is 10x+ slower!
    std::cout << "[Init] Running initial radiosity solve..." << std::endl;
    g_state.radiosity.runSolver(g_state.scene.h_primitives, g_state.scene.num_primitives,
                                g_state.scene.d_bvh_nodes, g_state.scene.d_bvh_indices,
                                g_state.config.enable_grid_filtering, g_state.config.use_bilateral_filter,
                                g_state.config.filter_sigma_spatial, g_state.config.filter_sigma_range);
    
    std::cout << "[Init] Precomputing CDFs for fast grid sampling..." << std::endl;
    g_state.scene.precomputeCDFs();
    
    // Copy updated primitives (with radiosity data) to device
    cudaMemcpy(g_state.scene.d_primitives, g_state.scene.h_primitives,
              g_state.scene.num_primitives * sizeof(Primitive), cudaMemcpyHostToDevice);
    
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
// RENDER FUNCTIONS
// ============================================================================

/**
 * @brief Render a single frame using path tracing or radiosity
 */
inline void renderFrame() {
    auto& profiler = Profiler::getInstance();
    
    // Update camera
    profiler.startStage("Camera Update");
    g_state.render.h_camera.updateCameraOrbit();
    cudaMemcpy(g_state.render.d_camera, &g_state.render.h_camera, sizeof(Sensor), cudaMemcpyHostToDevice);
    profiler.stopStage("Camera Update");
    
    // Reset kernel profile data if enabled
#if ENABLE_KERNEL_PROFILING
    if (g_state.config.enable_kernel_profiling && d_kernel_profile != nullptr) {
        h_kernel_profile.reset();
        cudaMemcpy(d_kernel_profile, &h_kernel_profile, sizeof(KernelProfileData), cudaMemcpyHostToDevice);
    }
#endif
    
    // Render kernel
    profiler.startStage("Render Kernel");
    switch (g_state.config.current_integrator) {
        case IntegratorType::PathTracing:
#if ENABLE_KERNEL_PROFILING
            if (g_state.config.enable_kernel_profiling && d_kernel_profile != nullptr) {
                render_profiled<<<g_state.render.grid, g_state.render.block>>>(
                    g_state.render.d_image, g_state.render.d_camera, g_state.scene.d_scene,
                    g_state.render.d_rand_state, g_state.config.spp, g_state.config.sampling_mode,
                    d_kernel_profile);
            } else
#endif
            {
                render<<<g_state.render.grid, g_state.render.block>>>(
                    g_state.render.d_image, g_state.render.d_camera, g_state.scene.d_scene,
                    g_state.render.d_rand_state, g_state.config.spp, g_state.config.sampling_mode);
            }
            break;

        case IntegratorType::Radiosity:
            render_radiosity<<<g_state.render.grid, g_state.render.block>>>(
                g_state.render.d_image, g_state.render.d_camera, g_state.scene.d_scene,
                g_state.render.d_rand_state, g_state.config.spp);
            break;
    }
    cudaDeviceSynchronize();
    profiler.stopStage("Render Kernel");
    
    // Copy kernel profile data back
#if ENABLE_KERNEL_PROFILING
    if (g_state.config.enable_kernel_profiling && d_kernel_profile != nullptr) {
        cudaMemcpy(&h_kernel_profile, d_kernel_profile, sizeof(KernelProfileData), cudaMemcpyDeviceToHost);
    }
#endif
    
    // Memory Transfer
    profiler.startStage("Memory Transfer");
    cudaMemcpy(g_state.render.h_image, g_state.render.d_image, g_state.render.img_size,
              cudaMemcpyDeviceToHost);
    profiler.stopStage("Memory Transfer");
    
    profiler.endFrame();
}

/**
 * @brief Render the frame to OpenGL quad
 */
inline void renderOpenGL() {
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
// CLEANUP
// ============================================================================

/**
 * @brief Cleanup all resources
 */
inline void cleanupApplication() {
    std::cout << "\nCleaning up resources..." << std::endl;
    
    g_state.render.cleanup();
    g_state.scene.cleanup();
    g_state.radiosity.cleanup();
    g_state.ui.cleanup();
    
#if ENABLE_KERNEL_PROFILING
    if (d_kernel_profile) cudaFree(d_kernel_profile);
#endif
}

/**
 * @brief Cleanup ImGui
 */
inline void cleanupImGui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

/**
 * @brief Cleanup GLFW
 */
inline void cleanupGLFW(GLFWwindow* window) {
    glfwDestroyWindow(window);
    glfwTerminate();
}

#endif // APPLICATION_H
