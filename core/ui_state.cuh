#ifndef UI_STATE_CUH
#define UI_STATE_CUH

#include "app_config.cuh"

// ============================================================================
// UI STATE MANAGEMENT
// ============================================================================

/**
 * @brief Manages UI state including mouse interaction and visualization
 */
struct UIState {
    // Mouse interaction
    bool is_dragging;
    double last_mouse_x;
    double last_mouse_y;
    
    // Window visibility
    bool show_grid_window;
    bool show_stats_window;
    bool show_optix_settings;
    
    // Selection state
    int hovered_primitive_idx;
    int selected_primitive_idx;
    
    // Grid visualization
    float* h_grid_data;
    GridVisualizationMode grid_viz_mode;
    
    // Performance stats
    float fps;
    float frame_time_ms;
    float render_time_ms;
    
    /**
     * @brief Constructor with default initialization
     */
    UIState() 
        : is_dragging(false)
        , last_mouse_x(0.0)
        , last_mouse_y(0.0)
        , show_grid_window(false)
        , show_stats_window(false)
        , show_optix_settings(false)
        , hovered_primitive_idx(-1)
        , selected_primitive_idx(-1)
        , h_grid_data(nullptr)
        , grid_viz_mode(GridVisualizationMode::VisibilityCount)
        , fps(0.0f)
        , frame_time_ms(0.0f)
        , render_time_ms(0.0f)
    {}
    
    /**
     * @brief Update performance statistics
     * @param delta_time Time since last frame
     */
    void updateStats(float delta_time);
    
    /**
     * @brief Allocate grid data buffer
     */
    void allocateGridData();
    
    /**
     * @brief Cleanup resources
     */
    void cleanup();
    
    /**
     * @brief Reset selection state
     */
    void resetSelection() {
        hovered_primitive_idx = -1;
        selected_primitive_idx = -1;
    }
};

#endif // UI_STATE_CUH
