/**
 * @file main.cu
 * @brief CUDA Path Tracer - Main Entry Point
 * 
 * This file serves as the entry point for the CUDA path tracer application.
 * Most functionality has been modularized into separate header files:
 * 
 * - cuda_utils.h         : CUDA utility functions (memory, error checking)
 * - application_state.h  : State structures (RenderState, SceneState, etc.)
 * - callbacks.h          : GLFW callbacks for input handling
 * - ui_windows.h         : ImGui UI window rendering
 * - application.h        : Initialization and render loop functions
 * - integrator.h         : Path tracing integrator and render kernels
 * - form_factors.h       : Radiosity form factor computation
 * - profiler.h           : Performance profiling
 * 
 * Core rendering files:
 * - scene.h              : Scene representation and ray intersection
 * - primitive.h          : Primitive types (Triangle, Quad)
 * - bvh.h                : BVH acceleration structure
 * - grid.h               : Grid-based importance sampling
 * - sensor.h             : Camera/sensor model
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"

// ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog.h"

// Application modules
#include "application_state.h"
#include "ui/callbacks.h"
#include "ui/ui_windows.h"
#include "application.h"

// ============================================================================
// GLOBAL STATE DEFINITION
// ============================================================================

// Global application state (declared extern in application_state.h)
ApplicationState g_state;

// Kernel profiling data (declared extern in application_state.h)
#if ENABLE_KERNEL_PROFILING
KernelProfileData* d_kernel_profile = nullptr;
KernelProfileData h_kernel_profile;
#endif

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv) {
    // Initialize OpenGL and GLFW
    GLFWwindow* window = nullptr;
    if (!initializeOpenGL(&window)) {
        return -1;
    }
    
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
        renderProfilerWindow();
        
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
    cleanupApplication();
    cleanupImGui();
    cleanupGLFW(window);
    
    std::cout << "Application terminated successfully" << std::endl;
    return 0;
}
