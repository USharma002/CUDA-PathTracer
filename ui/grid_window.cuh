#ifndef GRID_WINDOW_CUH
#define GRID_WINDOW_CUH

/**
 * @file grid_window.cuh
 * @brief Grid visualization window UI implementation
 * 
 * This file contains the ImGui-based grid visualization window for:
 * - Directional visibility grid visualization
 * - Radiosity distribution visualization
 * - Heatmap rendering with colormaps
 */

#include "../core/app_config.cuh"

// Forward declarations
struct RenderState;
struct SceneState;
struct UIState;

// ============================================================================
// GRID WINDOW CLASS
// ============================================================================

/**
 * @brief Grid visualization window for directional data
 */
class GridWindow {
public:
    /**
     * @brief Constructor
     */
    GridWindow();
    
    /**
     * @brief Destructor
     */
    ~GridWindow();
    
    /**
     * @brief Render the grid visualization window
     * 
     * @param scene Scene state reference
     * @param ui UI state reference
     */
    void render(SceneState& scene, UIState& ui);
    
    /**
     * @brief Set window visibility
     * @param visible Window visibility
     */
    void setVisible(bool visible) { visible_ = visible; }
    
    /**
     * @brief Check if window is visible
     * @return true if visible
     */
    bool isVisible() const { return visible_; }

private:
    bool visible_;
    
    // Colormap data
    float colormap_min_;
    float colormap_max_;
    bool auto_scale_;
    
    // Internal rendering methods
    void renderModeSelector(UIState& ui);
    void renderPrimitiveInfo(int prim_idx);
    void renderHeatmap(const float* grid_data, int width, int height, 
                       GridVisualizationMode mode);
    void renderColorbar(GridVisualizationMode mode);
    void renderLegend(GridVisualizationMode mode);
    
    // Colormap utilities
    void applyHotColormap(float value, float& r, float& g, float& b);
    void applyRadiosityColormap(float r_val, float g_val, float b_val,
                                 float& r, float& g, float& b, float max_val);
};

// ============================================================================
// COLORMAP UTILITIES
// ============================================================================

namespace colormap {

/**
 * @brief Hot colormap (black -> red -> orange -> yellow -> white)
 * 
 * @param value Normalized value (0-1)
 * @param r Output red component
 * @param g Output green component
 * @param b Output blue component
 */
inline void hot(float value, float& r, float& g, float& b) {
    if (value < 0.25f) {
        // Black to dark red
        r = value / 0.25f;
        g = 0.0f;
        b = 0.0f;
    } else if (value < 0.5f) {
        // Dark red to bright red
        r = 1.0f;
        g = (value - 0.25f) / 0.25f * 0.5f;
        b = 0.0f;
    } else if (value < 0.75f) {
        // Bright red to orange/yellow
        r = 1.0f;
        g = 0.5f + (value - 0.5f) / 0.25f * 0.5f;
        b = 0.0f;
    } else {
        // Yellow to white
        r = 1.0f;
        g = 1.0f;
        b = (value - 0.75f) / 0.25f;
    }
}

/**
 * @brief Viridis colormap (dark purple -> blue -> green -> yellow)
 * 
 * @param value Normalized value (0-1)
 * @param r Output red component
 * @param g Output green component
 * @param b Output blue component
 */
inline void viridis(float value, float& r, float& g, float& b) {
    // Simplified viridis approximation
    if (value < 0.25f) {
        r = 0.267f + value * 0.32f;
        g = 0.004f + value * 0.8f;
        b = 0.329f + value * 0.6f;
    } else if (value < 0.5f) {
        float t = (value - 0.25f) * 4.0f;
        r = 0.347f - t * 0.15f;
        g = 0.204f + t * 0.4f;
        b = 0.479f - t * 0.1f;
    } else if (value < 0.75f) {
        float t = (value - 0.5f) * 4.0f;
        r = 0.197f + t * 0.4f;
        g = 0.604f + t * 0.2f;
        b = 0.379f - t * 0.15f;
    } else {
        float t = (value - 0.75f) * 4.0f;
        r = 0.597f + t * 0.4f;
        g = 0.804f + t * 0.15f;
        b = 0.229f - t * 0.15f;
    }
}

/**
 * @brief Jet colormap (blue -> cyan -> green -> yellow -> red)
 * 
 * @param value Normalized value (0-1)
 * @param r Output red component
 * @param g Output green component
 * @param b Output blue component
 */
inline void jet(float value, float& r, float& g, float& b) {
    if (value < 0.25f) {
        r = 0.0f;
        g = value * 4.0f;
        b = 1.0f;
    } else if (value < 0.5f) {
        r = 0.0f;
        g = 1.0f;
        b = 1.0f - (value - 0.25f) * 4.0f;
    } else if (value < 0.75f) {
        r = (value - 0.5f) * 4.0f;
        g = 1.0f;
        b = 0.0f;
    } else {
        r = 1.0f;
        g = 1.0f - (value - 0.75f) * 4.0f;
        b = 0.0f;
    }
}

} // namespace colormap

// ============================================================================
// STANDALONE FUNCTIONS (for backward compatibility)
// ============================================================================

/**
 * @brief Render the grid visualization window (standalone function)
 */
void renderGridWindow();

/**
 * @brief Copy grid data from device to host
 * 
 * @param d_prims Device primitives
 * @param prim_idx Primitive index
 * @param h_grid Host grid buffer
 * @param num_prims Number of primitives
 * @param use_radiosity Use radiosity data instead of visibility
 */
void copyGridData(Primitive* d_prims, int prim_idx, float* h_grid,
                  int num_prims, bool use_radiosity = false);

#endif // GRID_WINDOW_CUH
