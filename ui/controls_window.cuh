#ifndef CONTROLS_WINDOW_CUH
#define CONTROLS_WINDOW_CUH

/**
 * @file controls_window.cuh
 * @brief Main controls window UI implementation
 * 
 * This file contains the ImGui-based controls window for:
 * - Resolution settings
 * - Rendering parameters
 * - Scene loading
 * - Integrator selection
 * - Radiosity controls
 */

// Forward declarations to avoid including ImGui headers everywhere
struct GLFWwindow;

// Include application state types
#include "../core/app_config.cuh"

// Forward declarations for state structures
struct RenderState;
struct SceneState;
struct RadiosityState;
struct UIState;
struct AppConfig;

// ============================================================================
// CONTROLS WINDOW CLASS
// ============================================================================

/**
 * @brief Main controls window for application settings
 */
class ControlsWindow {
public:
    /**
     * @brief Constructor
     */
    ControlsWindow();
    
    /**
     * @brief Destructor
     */
    ~ControlsWindow();
    
    /**
     * @brief Render the controls window
     * 
     * @param render Render state reference
     * @param scene Scene state reference
     * @param radiosity Radiosity state reference
     * @param ui UI state reference
     * @param config Application config reference
     */
    void render(RenderState& render, SceneState& scene, 
                RadiosityState& radiosity, UIState& ui, AppConfig& config);

private:
    // Window state
    bool show_advanced_options_;
    bool show_optix_options_;
    
    // Internal UI rendering methods
    void renderResolutionControls(RenderState& render);
    void renderRenderingControls(AppConfig& config);
    void renderSceneControls(SceneState& scene, RadiosityState& radiosity, 
                             AppConfig& config);
    void renderIntegratorControls(AppConfig& config);
    void renderRadiosityControls(RadiosityState& radiosity, SceneState& scene);
    void renderHistoryControls(AppConfig& config);
    void renderOptixControls(AppConfig& config, SceneState& scene);
    void renderSaveControls(RenderState& render);
    void renderStatistics(SceneState& scene, RadiosityState& radiosity);
    
    // File dialogs
    void handleFileDialogs(SceneState& scene, RadiosityState& radiosity,
                           RenderState& render, AppConfig& config);
};

// ============================================================================
// STANDALONE FUNCTIONS (for backward compatibility)
// ============================================================================

/**
 * @brief Render the main controls window (standalone function)
 * 
 * This function provides backward compatibility with the original implementation.
 */
void renderControlsWindow();

#endif // CONTROLS_WINDOW_CUH
