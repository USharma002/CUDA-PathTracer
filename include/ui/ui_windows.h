/**
 * @file ui_windows.h
 * @brief ImGui window rendering functions
 * 
 * Contains all UI window rendering code including controls, grid visualization,
 * and profiler windows.
 */

#ifndef UI_WINDOWS_H
#define UI_WINDOWS_H

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog.h"

// Note: stb_image_write.h is included in main.cu with STB_IMAGE_WRITE_IMPLEMENTATION
// We only declare the function we need here
extern "C" int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
extern "C" void stbi_flip_vertically_on_write(int flag);

#include "application_state.h"
#include "utils/profiler.h"

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Copy grid data from device primitive to host buffer
 */
inline void copyGridData(Primitive* d_prims, int prim_idx, float* h_grid, 
                         int num_prims, bool use_radiosity = false) {
    if (prim_idx < 0 || prim_idx >= num_prims || !d_prims) {
        memset(h_grid, 0, GRID_SIZE * 3 * sizeof(float));
        return;
    }
    
    Primitive temp_prim;
    cudaMemcpy(&temp_prim, &d_prims[prim_idx], sizeof(Primitive), cudaMemcpyDeviceToHost);
    
    if (use_radiosity) {
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
// CONTROLS WINDOW
// ============================================================================

inline void renderControlsWindow() {
    ImGui::Begin("Controls");
    
    bool res_changed = false;
    res_changed |= ImGui::SliderInt("Width", &g_state.render.width, 200, 2000);
    res_changed |= ImGui::SliderInt("Height", &g_state.render.height, 200, 2000);
    
    if (res_changed) {
        g_state.render.allocateBuffers();
    }
    
    ImGui::SliderInt("SPP", &g_state.config.spp, 1, 1000);
    
    // Scene loading
    if (ImGui::Button("Browse Scene")) {
        IGFD::FileDialogConfig config;
        config.path = ".";
        // Support both OBJ and PBRT formats
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose Scene File", ".obj,.pbrt", config);
    }
    
    if (ImGui::Checkbox("Convert Quads to Triangles", &g_state.config.convert_quads_to_triangles)) {
        g_state.scene.loadScene(g_state.scene.scene_file, g_state.config.subdivision_step, 
                                g_state.config.convert_quads_to_triangles);
        g_state.radiosity.runSolver(g_state.scene.h_primitives, g_state.scene.num_primitives,
                                    g_state.scene.d_bvh_nodes, g_state.scene.d_bvh_indices,
                                    g_state.config.enable_grid_filtering, g_state.config.use_bilateral_filter,
                                    g_state.config.filter_sigma_spatial, g_state.config.filter_sigma_range);
        g_state.scene.precomputeCDFs();
        cudaMemcpy(g_state.scene.d_primitives, g_state.scene.h_primitives,
                  g_state.scene.num_primitives * sizeof(Primitive), cudaMemcpyHostToDevice);
    }

    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            g_state.scene.scene_file = ImGuiFileDialog::Instance()->GetFilePathName();
            g_state.scene.loadScene(g_state.scene.scene_file, g_state.config.subdivision_step,
                                    g_state.config.convert_quads_to_triangles);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    const char* sampling_items[] = { "BSDF Sampling", "Grid Sampling", "MIS (Mixed)" };
    static int current_sampling = 1;
    if (ImGui::Combo("Sampling Mode", &current_sampling, sampling_items, 3)) {
        g_state.config.sampling_mode = (SamplingMode)current_sampling;
    }
    
    if (g_state.config.sampling_mode == SamplingMode::SAMPLING_MIS) {
        if (ImGui::SliderFloat("BSDF Fraction", &g_state.config.mis_bsdf_fraction, 0.0f, 1.0f, "%.2f")) {
            g_state.scene.setMisBsdfFraction(g_state.config.mis_bsdf_fraction);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("MIS BSDF fraction:\n"
                              "0.0 = Pure Grid sampling\n"
                              "0.5 = 50/50 mix (default)\n"
                              "1.0 = Pure BSDF sampling");
        }
    }

    const char* integrator_names[] = { "Path Tracing", "Radiosity" };
    int current_integrator = (int)g_state.config.current_integrator;
    if (ImGui::Combo("Integrator", &current_integrator, integrator_names, 2)) {
        g_state.config.current_integrator = (IntegratorType)current_integrator;
    }

    bool rad_changed = ImGui::SliderInt("Radiosity Steps", &g_state.radiosity.num_iterations, 0, 50);
    ImGui::Checkbox("Use Monte Carlo", &g_state.radiosity.use_monte_carlo);
    if (g_state.radiosity.use_monte_carlo) {
        ImGui::SliderInt("MC Samples", &g_state.radiosity.mc_samples, 4, 256);
    }
    
    ImGui::Separator();
    ImGui::Text("Grid Filtering (16x16 = 256 cells):");
    
    ImGui::Checkbox("Bilateral (vs Gaussian)", &g_state.config.use_bilateral_filter);
    ImGui::SliderFloat("Spatial Sigma", &g_state.config.filter_sigma_spatial, 0.5f, 5.0f);
    if (g_state.config.use_bilateral_filter) {
        ImGui::SliderFloat("Range Sigma", &g_state.config.filter_sigma_range, 0.05f, 1.0f);
    }
    
    if (ImGui::Button("Apply Filter & Rebuild CDFs")) {
        std::cout << "\n[Filter] Applying " << (g_state.config.use_bilateral_filter ? "bilateral" : "gaussian") 
                  << " filter..." << std::endl;
        filter_pdfs_for_primitives(g_state.scene.d_primitives,
                                   g_state.scene.d_filtered_formfactor,
                                   g_state.scene.d_filtered_radiosity,
                                   g_state.scene.num_primitives,
                                   GRID_RES,
                                   g_state.config.use_bilateral_filter,
                                   g_state.config.filter_sigma_spatial,
                                   g_state.config.filter_sigma_range);
        g_state.scene.precomputeCDFsFromFiltered();
        std::cout << "[Filter] Complete - using FILTERED CDFs for sampling" << std::endl;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Apply bilateral/gaussian filtering and precompute CDFs.\n"
                          "CDFs are always precomputed for fast sampling!");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Use Raw CDFs")) {
        g_state.scene.precomputeCDFs();
        std::cout << "[Filter] Switched to RAW radiosity CDFs" << std::endl;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Rebuild CDFs from raw (unfiltered) radiosity.\n"
                          "Use this if you want to sample without filtering.");
    }
    ImGui::Separator();
    
    if (ImGui::Button("Calculate Radiosity") || rad_changed) {
        g_state.radiosity.runSolver(g_state.scene.h_primitives, g_state.scene.num_primitives,
                                    g_state.scene.d_bvh_nodes, g_state.scene.d_bvh_indices,
                                    g_state.config.enable_grid_filtering, g_state.config.use_bilateral_filter,
                                    g_state.config.filter_sigma_spatial, g_state.config.filter_sigma_range);
        g_state.scene.precomputeCDFs();
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
        g_state.scene.loadScene(g_state.scene.scene_file, g_state.config.subdivision_step,
                                g_state.config.convert_quads_to_triangles);
        g_state.radiosity.runSolver(g_state.scene.h_primitives, g_state.scene.num_primitives,
                                    g_state.scene.d_bvh_nodes, g_state.scene.d_bvh_indices,
                                    g_state.config.enable_grid_filtering, g_state.config.use_bilateral_filter,
                                    g_state.config.filter_sigma_spatial, g_state.config.filter_sigma_range);
        g_state.scene.precomputeCDFs();
        cudaMemcpy(g_state.scene.d_primitives, g_state.scene.h_primitives,
                  g_state.scene.num_primitives * sizeof(Primitive), cudaMemcpyHostToDevice);
    }
    
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

// ============================================================================
// GRID VISUALIZATION WINDOW
// ============================================================================

inline void renderGridWindow() {
    if (!g_state.ui.show_grid_window) return;
    
    ImGui::Begin("Sampling PDF Visualization", &g_state.ui.show_grid_window);
    
    const char* mode_names[] = { "BSDF", "Grid", "Radiosity", "MIS", "TopK" };
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), 
                       "Sampling Mode: %s", mode_names[(int)g_state.config.sampling_mode]);
    ImGui::Text("Grid: %d x %d cells", GRID_RES, GRID_RES);
    
    bool using_filtered = g_state.scene.h_scene.use_filtered && g_state.scene.d_filtered_radiosity != nullptr;
    if (using_filtered) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "Source: FILTERED PDF");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Source: RAW Radiosity Grid");
    }
    
    if (g_state.ui.hovered_primitive_idx >= 0 && 
        g_state.ui.hovered_primitive_idx < g_state.scene.num_primitives) {
        
        ImGui::Separator();
        ImGui::Text("Primitive: %d", g_state.ui.hovered_primitive_idx);
        
        if (!g_state.ui.h_grid_data) {
            g_state.ui.h_grid_data = new float[GRID_SIZE];
        }
        
        float max_val = 0.0f, total_sum = 0.0f;
        int non_zero = 0;
        
        if (using_filtered) {
            size_t offset = (size_t)g_state.ui.hovered_primitive_idx * GRID_SIZE;
            cudaMemcpy(g_state.ui.h_grid_data, 
                       g_state.scene.d_filtered_radiosity + offset,
                       GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < GRID_SIZE; i++) {
                float val = g_state.ui.h_grid_data[i];
                max_val = fmaxf(max_val, val);
                total_sum += val;
                if (val > 1e-6f) non_zero++;
            }
        } else {
            Primitive temp_prim;
            cudaMemcpy(&temp_prim, &g_state.scene.d_primitives[g_state.ui.hovered_primitive_idx], 
                       sizeof(Primitive), cudaMemcpyDeviceToHost);
            
            Vector3f* rad_grid = (temp_prim.type == PRIM_TRIANGLE) ? 
                                  temp_prim.tri.radiosity_grid : temp_prim.quad.radiosity_grid;
            
            for (int i = 0; i < GRID_SIZE; i++) {
                float lum = 0.2126f * rad_grid[i].x() + 
                            0.7152f * rad_grid[i].y() + 
                            0.0722f * rad_grid[i].z();
                g_state.ui.h_grid_data[i] = lum;
                max_val = fmaxf(max_val, lum);
                total_sum += lum;
                if (lum > 1e-6f) non_zero++;
            }
        }
        
        ImGui::Text("Max: %.4f | Sum: %.4f | Non-zero: %d", max_val, total_sum, non_zero);
        
        ImVec2 size(300, 300);
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        
        float cell_w = size.x / GRID_RES;
        float cell_h = size.y / GRID_RES;
        
        for (int row = 0; row < GRID_RES; row++) {
            for (int col = 0; col < GRID_RES; col++) {
                int idx = row * GRID_RES + col;
                float val = (max_val > 1e-6f) ? fminf(g_state.ui.h_grid_data[idx] / max_val, 1.0f) : 0.0f;
                
                float r, g, b;
                if (val < 0.33f) {
                    r = val * 3.0f; g = 0.0f; b = 0.0f;
                } else if (val < 0.67f) {
                    r = 1.0f; g = (val - 0.33f) * 3.0f; b = 0.0f;
                } else {
                    r = 1.0f; g = 1.0f; b = (val - 0.67f) * 3.0f;
                }
                
                ImU32 color = IM_COL32((int)(r * 255), (int)(g * 255), (int)(b * 255), 255);
                ImVec2 p_min(canvas_pos.x + col * cell_w, canvas_pos.y + row * cell_h);
                ImVec2 p_max(p_min.x + cell_w, p_min.y + cell_h);
                draw_list->AddRectFilled(p_min, p_max, color);
            }
        }
        
        ImGui::Dummy(size);
        ImGui::Text("Shows: %s (exact sampling PDF)", using_filtered ? "Filtered PDF" : "Raw Radiosity Luminance");
    } else {
        ImGui::Text("Hover over a primitive to see its sampling PDF");
    }
    
    ImGui::End();
}

// ============================================================================
// PROFILER WINDOW
// ============================================================================

inline void renderProfilerWindow() {
    auto& profiler = Profiler::getInstance();
    
    ImGui::Begin("Profiler", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    float fps = profiler.getFPS();
    float frame_time = profiler.getTotalFrameMs();
    float avg_frame_time = profiler.getAverageFrameTime();
    
    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Frame Time: %.2f ms (avg: %.2f ms)", frame_time, avg_frame_time);
    ImGui::Separator();
    
    bool enabled = profiler.isEnabled();
    if (ImGui::Checkbox("Enable Profiling", &enabled)) {
        profiler.setEnabled(enabled);
    }
    
    if (ImGui::Button("Reset Stats")) {
        profiler.reset();
    }
    
    ImGui::Separator();
    
    ImGui::Text("Stage Breakdown:");
    const auto& stages = profiler.getStages();
    
    float total_ms = 0.0f;
    for (const auto& stage : stages) {
        total_ms += stage.current_ms;
    }
    
    if (total_ms > 0.0f && !stages.empty()) {
        ImVec2 bar_size(300, 20);
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        ImU32 colors[] = {
            IM_COL32(66, 133, 244, 255),
            IM_COL32(219, 68, 55, 255),
            IM_COL32(244, 180, 0, 255),
            IM_COL32(15, 157, 88, 255),
            IM_COL32(171, 71, 188, 255),
            IM_COL32(255, 112, 67, 255),
        };
        int num_colors = sizeof(colors) / sizeof(colors[0]);
        
        float x_offset = 0.0f;
        int idx = 0;
        for (const auto& stage : stages) {
            float width = (stage.current_ms / total_ms) * bar_size.x;
            ImVec2 p_min(p0.x + x_offset, p0.y);
            ImVec2 p_max(p0.x + x_offset + width, p0.y + bar_size.y);
            draw_list->AddRectFilled(p_min, p_max, colors[idx % num_colors]);
            x_offset += width;
            idx++;
        }
        
        draw_list->AddRect(p0, ImVec2(p0.x + bar_size.x, p0.y + bar_size.y), IM_COL32(255, 255, 255, 128));
        ImGui::Dummy(bar_size);
        
        idx = 0;
        for (const auto& stage : stages) {
            ImU32 color = colors[idx % num_colors];
            ImVec2 legend_pos = ImGui::GetCursorScreenPos();
            draw_list->AddRectFilled(legend_pos, ImVec2(legend_pos.x + 12, legend_pos.y + 12), color);
            ImGui::Dummy(ImVec2(14, 12));
            ImGui::SameLine();
            ImGui::Text("%s: %.2f ms (%.1f%%)", stage.name.c_str(), stage.current_ms, 
                       (stage.current_ms / total_ms) * 100.0f);
            idx++;
        }
    }
    
    ImGui::Separator();
    
    ImGui::Text("Frame Time History:");
    const float* history = profiler.getFrameHistory();
    
    float max_time = 1.0f;
    for (int i = 0; i < PROFILER_HISTORY_SIZE; i++) {
        max_time = fmaxf(max_time, history[i]);
    }
    
    ImVec2 hist_size(300, 80);
    ImVec2 h0 = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    
    draw_list->AddRectFilled(h0, ImVec2(h0.x + hist_size.x, h0.y + hist_size.y), IM_COL32(30, 30, 30, 255));
    
    float y_60fps = h0.y + hist_size.y - (16.67f / max_time) * hist_size.y;
    float y_30fps = h0.y + hist_size.y - (33.33f / max_time) * hist_size.y;
    
    if (16.67f < max_time) {
        draw_list->AddLine(ImVec2(h0.x, y_60fps), ImVec2(h0.x + hist_size.x, y_60fps), 
                          IM_COL32(0, 255, 0, 100), 1.0f);
    }
    if (33.33f < max_time) {
        draw_list->AddLine(ImVec2(h0.x, y_30fps), ImVec2(h0.x + hist_size.x, y_30fps), 
                          IM_COL32(255, 255, 0, 100), 1.0f);
    }
    
    float bar_width = hist_size.x / PROFILER_HISTORY_SIZE;
    int start_idx = profiler.getFrameHistoryIdx();
    
    for (int i = 0; i < PROFILER_HISTORY_SIZE; i++) {
        int idx = (start_idx + i) % PROFILER_HISTORY_SIZE;
        float val = history[idx];
        if (val > 0.0f) {
            float bar_height = (val / max_time) * hist_size.y;
            ImVec2 p_min(h0.x + i * bar_width, h0.y + hist_size.y - bar_height);
            ImVec2 p_max(h0.x + (i + 1) * bar_width - 1, h0.y + hist_size.y);
            
            ImU32 color;
            if (val < 16.67f) {
                color = IM_COL32(0, 200, 0, 255);
            } else if (val < 33.33f) {
                color = IM_COL32(200, 200, 0, 255);
            } else {
                color = IM_COL32(200, 0, 0, 255);
            }
            draw_list->AddRectFilled(p_min, p_max, color);
        }
    }
    
    draw_list->AddRect(h0, ImVec2(h0.x + hist_size.x, h0.y + hist_size.y), IM_COL32(100, 100, 100, 255));
    ImGui::Dummy(hist_size);
    
    ImGui::Text("Green: <16ms (60fps) | Yellow: <33ms (30fps) | Red: >33ms");
    ImGui::Text("Max: %.1f ms | Frames: %d", max_time, profiler.getFrameCount());
    
#if ENABLE_KERNEL_PROFILING
    ImGui::Separator();
    ImGui::Checkbox("Enable Kernel Profiling", &g_state.config.enable_kernel_profiling);
    
    if (g_state.config.enable_kernel_profiling && h_kernel_profile.total_samples > 0) {
        ImGui::Text("Kernel Breakdown (GPU cycles):");
        
        unsigned long long total_cycles = h_kernel_profile.intersection_cycles + 
                                          h_kernel_profile.grid_init_cycles +
                                          h_kernel_profile.sampling_cycles +
                                          h_kernel_profile.shading_cycles;
        
        if (total_cycles > 0) {
            float pct_intersect = 100.0f * h_kernel_profile.intersection_cycles / total_cycles;
            float pct_grid_init = 100.0f * h_kernel_profile.grid_init_cycles / total_cycles;
            float pct_sampling = 100.0f * h_kernel_profile.sampling_cycles / total_cycles;
            float pct_shading = 100.0f * h_kernel_profile.shading_cycles / total_cycles;
            
            ImVec2 bar_size2(300, 20);
            ImVec2 p1 = ImGui::GetCursorScreenPos();
            
            float x = 0.0f;
            float w_int = (pct_intersect / 100.0f) * bar_size2.x;
            draw_list->AddRectFilled(ImVec2(p1.x + x, p1.y), ImVec2(p1.x + x + w_int, p1.y + bar_size2.y), 
                                    IM_COL32(66, 133, 244, 255));
            x += w_int;
            
            float w_grid = (pct_grid_init / 100.0f) * bar_size2.x;
            draw_list->AddRectFilled(ImVec2(p1.x + x, p1.y), ImVec2(p1.x + x + w_grid, p1.y + bar_size2.y), 
                                    IM_COL32(219, 68, 55, 255));
            x += w_grid;
            
            float w_samp = (pct_sampling / 100.0f) * bar_size2.x;
            draw_list->AddRectFilled(ImVec2(p1.x + x, p1.y), ImVec2(p1.x + x + w_samp, p1.y + bar_size2.y), 
                                    IM_COL32(244, 180, 0, 255));
            x += w_samp;
            
            float w_shade = (pct_shading / 100.0f) * bar_size2.x;
            draw_list->AddRectFilled(ImVec2(p1.x + x, p1.y), ImVec2(p1.x + x + w_shade, p1.y + bar_size2.y), 
                                    IM_COL32(15, 157, 88, 255));
            
            draw_list->AddRect(p1, ImVec2(p1.x + bar_size2.x, p1.y + bar_size2.y), IM_COL32(255, 255, 255, 128));
            ImGui::Dummy(bar_size2);
            
            ImGui::TextColored(ImVec4(0.26f, 0.52f, 0.96f, 1.0f), "Ray Intersect: %.1f%%", pct_intersect);
            ImGui::TextColored(ImVec4(0.86f, 0.27f, 0.22f, 1.0f), "Grid Init: %.1f%% (CDFs)", pct_grid_init);
            ImGui::TextColored(ImVec4(0.96f, 0.71f, 0.0f, 1.0f), "Sampling: %.1f%%", pct_sampling);
            ImGui::TextColored(ImVec4(0.06f, 0.62f, 0.35f, 1.0f), "Shading: %.1f%%", pct_shading);
            
            ImGui::Separator();
            ImGui::Text("Samples: %llu | Grid samples: %llu", 
                       h_kernel_profile.total_samples, h_kernel_profile.grid_samples);
            
            if (h_kernel_profile.total_samples > 0) {
                float avg_intersect = (float)h_kernel_profile.intersection_cycles / h_kernel_profile.total_samples;
                float avg_grid = h_kernel_profile.grid_samples > 0 ? 
                    (float)h_kernel_profile.grid_init_cycles / h_kernel_profile.grid_samples : 0;
                ImGui::Text("Avg cycles/sample: Intersect=%.0f, Grid=%.0f", avg_intersect, avg_grid);
            }
        }
    } else if (g_state.config.enable_kernel_profiling) {
        ImGui::Text("Kernel profiling: waiting for data...");
    }
#endif
    
    ImGui::End();
}

#endif // UI_WINDOWS_H
