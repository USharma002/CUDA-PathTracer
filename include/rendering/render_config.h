#ifndef RENDER_CONFIG_H
#define RENDER_CONFIG_H

// ============================================================================
// GRID CONFIGURATION - Single source of truth for all grid parameters
// ============================================================================
#define GRID_RES 16              // Grid resolution (16x16 = 256 cells for speed)
#define GRID_SIZE (GRID_RES * GRID_RES)  // Total number of grid cells (256)
#define GRID_HALF_RES (GRID_RES / 2)     // Half resolution for hemisphere (8)
#define MIN_GRID_RES 8
#define MAX_GRID_RES 64

// Precomputed constants for grid sampling (avoid repeated divisions)
#define GRID_INV_RES (1.0f / GRID_RES)
#define GRID_INV_HALF_RES (1.0f / GRID_HALF_RES)
#define GRID_D_THETA ((M_PI * 0.5f) / GRID_HALF_RES)  // Theta step for hemisphere
#define GRID_D_PHI (2.0f * M_PI / GRID_RES)           // Phi step

// ============================================================================
// PRE-COMPUTED CDF DATA STRUCTURE (OPTIMIZED FOR 16x16 = 256 cells)
// This stores all CDF data for one primitive, pre-computed on CPU
// Size: 256*4 + 8*4 + 8*4 + 256*4 + 4 + 4 = ~2KB per primitive (cache-friendly!)
// ============================================================================
struct PrecomputedCDF {
    float pdf[GRID_SIZE];              // The PDF values (luminance) - 256 floats
    float row_sums[GRID_HALF_RES];     // Sum of each row (upper hemisphere only) - 8 floats
    float marginal_cdf[GRID_HALF_RES]; // CDF for row selection - 8 floats
    float row_cdfs[GRID_SIZE];         // Pre-built conditional CDFs for all rows - 256 floats
    float total_weight;                // Sum of all pdf values in upper hemisphere
    int is_valid;                      // 1 if valid, 0 otherwise (use int for alignment)
};

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

// ==================== KERNEL PROFILING ====================
// Enable/disable detailed kernel profiling (adds overhead)
#define ENABLE_KERNEL_PROFILING 1

#if ENABLE_KERNEL_PROFILING
// Profiling counters (updated atomically by kernel threads)
struct KernelProfileData {
    unsigned long long intersection_cycles;   // Ray-scene intersection
    unsigned long long grid_init_cycles;      // Grid initialization (copy + CDF build)
    unsigned long long sampling_cycles;       // Direction sampling
    unsigned long long shading_cycles;        // BSDF evaluation
    unsigned long long total_samples;         // Number of samples processed
    unsigned long long grid_samples;          // Number of grid samples taken
    
    __host__ __device__ void reset() {
        intersection_cycles = 0;
        grid_init_cycles = 0;
        sampling_cycles = 0;
        shading_cycles = 0;
        total_samples = 0;
        grid_samples = 0;
    }
};
#endif

#endif // RENDER_CONFIG_H