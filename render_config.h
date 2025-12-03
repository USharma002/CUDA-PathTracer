#ifndef RENDER_CONFIG_H
#define RENDER_CONFIG_H

// ============================================================================
// GRID CONFIGURATION - Single source of truth for all grid parameters
// ============================================================================
#define GRID_RES 50              // Grid resolution (GRID_RES x GRID_RES cells)
#define GRID_SIZE (GRID_RES * GRID_RES)  // Total number of grid cells
#define MIN_GRID_RES 10
#define MAX_GRID_RES 200

// Aliases for backward compatibility
#define DEFAULT_GRID_RES GRID_RES
#define GRID_RESOLUTION GRID_RES

// ==================== SAMPLING CONFIGURATION ====================
enum class SamplingMode {
    SAMPLING_BSDF = 0,        // Pure BSDF sampling
    SAMPLING_FORMFACTOR = 1,  // Form factor grid
    SAMPLING_RADIOSITY = 2,   // Radiosity grid
    SAMPLING_MIS = 3,         // Multiple Importance Sampling
    SAMPLING_TOPK = 4         // Top-K cells only
};

// ==================== TOP-K CONFIGURATION ====================
#define MAX_TOP_K 1000
#define DEFAULT_TOP_K 0  // 0 = use all cells

// ==================== PERFORMANCE TUNING ====================
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

#endif // RENDER_CONFIG_H