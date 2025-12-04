/**
 * @file callbacks.h
 * @brief GLFW callbacks and input handling
 * 
 * Contains all GLFW callback functions for window, mouse, and keyboard events.
 */

#ifndef CALLBACKS_H
#define CALLBACKS_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include "imgui.h"
#include "application_state.h"

// ============================================================================
// PRIMITIVE PICKING KERNEL
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
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Pick primitive under mouse cursor using GPU ray casting
 */
inline void pickPrimitive(int mouse_x, int mouse_y, int width, int height,
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

// ============================================================================
// GLFW CALLBACKS
// ============================================================================

/**
 * @brief Framebuffer size change callback
 */
inline void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    if (width > 0 && height > 0) {
        g_state.render.updateResolution(width, height);
        glViewport(0, 0, width, height);
    }
}

/**
 * @brief Mouse button callback for camera rotation
 */
inline void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && !ImGui::GetIO().WantCaptureMouse) {
        g_state.ui.is_dragging = (action == GLFW_PRESS);
        if (g_state.ui.is_dragging) {
            glfwGetCursorPos(window, &g_state.ui.last_mouse_x, &g_state.ui.last_mouse_y);
        }
    }
}

/**
 * @brief Cursor position callback for camera control and primitive picking
 */
inline void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
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

/**
 * @brief Scroll callback for camera zoom
 */
inline void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (!ImGui::GetIO().WantCaptureMouse) {
        g_state.render.h_camera.radius -= yoffset * ZOOM_SENSITIVITY;
    }
}

/**
 * @brief Register all GLFW callbacks
 */
inline void registerCallbacks(GLFWwindow* window) {
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
}

#endif // CALLBACKS_H
