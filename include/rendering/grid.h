/**
 * @file grid.h
 * @brief Highly optimized Grid class for direction sampling in path tracing
 * 
 * OPTIMIZED FOR 16x16 GRID:
 * - ZERO-COPY: Uses direct pointer to precomputed CDFs (no memory copy!)
 * - Linear search for small arrays (faster than binary search for N<=16)
 * - Precomputed constants to avoid divisions
 * - Cache-friendly memory layout
 * 
 * Design: The integrator just calls grid.sample() - all details are hidden here.
 */

#ifndef GRID_H
#define GRID_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "core/vector.h"
#include "rendering/render_config.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================================
// GRID CLASS - ZERO-COPY version using direct pointer to precomputed CDFs
// ============================================================================

class Grid {
public:
    // ZERO-COPY: Direct pointer to precomputed data (no copying!)
    const PrecomputedCDF* precomp_ptr;       // Pointer to precomputed data
    
    // Fallback storage for legacy init (rarely used on GPU)
    float pdf[GRID_SIZE];                    // PDF values (luminance) - 256 floats
    float row_sums[GRID_HALF_RES];           // Sum of each row (upper hemisphere) - 8 floats
    float marginal_cdf[GRID_HALF_RES];       // CDF for row selection - 8 floats
    float row_cdfs[GRID_SIZE];               // Pre-built conditional CDFs - 256 floats
    float total_weight;                       // Total PDF weight
    bool is_valid;                            // Valid flag
    bool use_precomp;                         // True if using precomputed pointer (fast path)
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    __host__ __device__ Grid() : precomp_ptr(nullptr), total_weight(0.0f), is_valid(false), use_precomp(false) {}
    
    // Legacy: Initialize from float array (form factors)
    __host__ __device__ void initFromFormFactor(const float* ff_grid) {
        use_precomp = false;
        precomp_ptr = nullptr;
        #pragma unroll
        for (int i = 0; i < GRID_SIZE; i++) {
            pdf[i] = ff_grid[i];
        }
        buildCDFs();
    }
    
    // Initialize from Vector3f array (radiosity) - converts to luminance
    __host__ __device__ void initFromRadiosity(const Vector3f* rad_grid) {
        use_precomp = false;
        precomp_ptr = nullptr;
        #pragma unroll 4
        for (int i = 0; i < GRID_SIZE; i++) {
            // ITU-R BT.709 luminance (simplified)
            pdf[i] = 0.2126f * rad_grid[i].x() + 
                     0.7152f * rad_grid[i].y() + 
                     0.0722f * rad_grid[i].z();
        }
        buildCDFs();
    }
    
    // ========================================================================
    // ZERO-COPY LOAD FROM PRECOMPUTED CDFs (PRIMARY PATH - INSTANT!)
    // This is the ONLY recommended way to initialize for sampling
    // Just stores a pointer - NO data copying at all!
    // ========================================================================
    __device__ __forceinline__ void loadPrecomputed(const PrecomputedCDF* precomp) {
        precomp_ptr = precomp;
        use_precomp = true;
        is_valid = (precomp->is_valid != 0);
        total_weight = precomp->total_weight;  // Cache this one value for quick checks
    }
    
    // Build ALL CDFs upfront for fast sampling (CPU only, avoid on GPU!)
    __host__ __device__ void buildCDFs() {
        total_weight = 0.0f;
        
        // Compute row sums for upper hemisphere only
        for (int v = 0; v < GRID_HALF_RES; v++) {
            float row_sum = 0.0f;
            const int row_offset = v * GRID_RES;
            
            for (int u = 0; u < GRID_RES; u++) {
                row_sum += pdf[row_offset + u];
            }
            row_sums[v] = row_sum;
            total_weight += row_sum;
        }
        
        // Build marginal CDF
        float running = 0.0f;
        const float inv_total = (total_weight > 1e-6f) ? (1.0f / total_weight) : 0.0f;
        for (int v = 0; v < GRID_HALF_RES; v++) {
            running += row_sums[v];
            marginal_cdf[v] = running * inv_total;
        }
        if (GRID_HALF_RES > 0) marginal_cdf[GRID_HALF_RES - 1] = 1.0f;
        
        // PRE-BUILD all row CDFs for fast sampling
        for (int v = 0; v < GRID_HALF_RES; v++) {
            const int row_offset = v * GRID_RES;
            float row_sum = row_sums[v];
            
            if (row_sum < 1e-6f) {
                // Uniform distribution
                for (int u = 0; u < GRID_RES; u++) {
                    row_cdfs[row_offset + u] = (u + 1) * GRID_INV_RES;
                }
            } else {
                float running_row = 0.0f;
                const float inv_row_sum = 1.0f / row_sum;
                for (int u = 0; u < GRID_RES; u++) {
                    running_row += pdf[row_offset + u];
                    row_cdfs[row_offset + u] = running_row * inv_row_sum;
                }
                row_cdfs[row_offset + GRID_RES - 1] = 1.0f;
            }
        }
        
        is_valid = (total_weight > 1e-6f);
    }
    
    // ========================================================================
    // OPTIMIZED SAMPLING (for 16x16 grid with precomputed CDFs)
    // Uses direct pointer access - no copy!
    // ========================================================================
    
    __device__ __forceinline__ Vector3f sample(const Vector3f& normal, curandState* rng, float& out_pdf) const {
        if (!is_valid) {
            return sampleCosineHemisphere(normal, rng, out_pdf);
        }
        
        // Generate random numbers
        float xi1 = curand_uniform(rng);
        float xi2 = curand_uniform(rng);
        
        // Get pointers to CDF data (either precomputed or local)
        const float* marg_cdf = use_precomp ? precomp_ptr->marginal_cdf : marginal_cdf;
        const float* r_cdfs = use_precomp ? precomp_ptr->row_cdfs : row_cdfs;
        
        // Linear search for row (theta) - optimal for small N
        int theta_idx = linearSearchCDF(marg_cdf, GRID_HALF_RES, xi1);
        
        // Linear search for column (phi) in selected row
        const float* row_cdf = &r_cdfs[theta_idx * GRID_RES];
        int phi_idx = linearSearchCDF(row_cdf, GRID_RES, xi2);
        
        // Convert to direction with jitter
        float jitter_theta = curand_uniform(rng);
        float jitter_phi = curand_uniform(rng);
        
        // Spherical coordinates using precomputed constants
        float theta = (theta_idx + jitter_theta) * GRID_INV_HALF_RES * (M_PI * 0.5f);
        theta = fminf(theta, M_PI * 0.5f - 0.01f);
        float phi = (phi_idx + jitter_phi) * GRID_INV_RES * 2.0f * M_PI;
        
        // Use sincos for efficiency
        float sin_t, cos_t, sin_p, cos_p;
        sincosf(theta, &sin_t, &cos_t);
        sincosf(phi, &sin_p, &cos_p);
        
        Vector3f local_dir(sin_t * cos_p, sin_t * sin_p, cos_t);
        
        // Transform to world coordinates
        Vector3f tangent, bitangent;
        buildFrame(normal, tangent, bitangent);
        Vector3f world_dir = unit_vector(tangent * local_dir.x() + 
                                          bitangent * local_dir.y() + 
                                          normal * local_dir.z());
        
        // Compute PDF inline
        out_pdf = computePDFForCell(theta_idx, phi_idx);
        
        return world_dir;
    }
    
    // Simple sample without PDF
    __device__ __forceinline__ Vector3f sample(const Vector3f& normal, curandState* rng) const {
        float unused;
        return sample(normal, rng, unused);
    }
    
    // ========================================================================
    // PDF COMPUTATION
    // ========================================================================
    
    __device__ float computePDF(const Vector3f& dir, const Vector3f& normal) const {
        if (!is_valid) return cosinePDF(dir, normal);
        
        // World to local spherical
        float theta, phi;
        worldToSpherical(dir, normal, theta, phi);
        
        if (theta > M_PI * 0.5f) return 0.0f;
        
        // Map to grid indices using precomputed constants
        int theta_idx = (int)(theta * (2.0f / M_PI) * GRID_HALF_RES);
        int phi_idx = (int)(phi * (0.5f / M_PI) * GRID_RES);
        theta_idx = max(0, min(theta_idx, GRID_HALF_RES - 1));
        phi_idx = max(0, min(phi_idx, GRID_RES - 1));
        
        return computePDFForCell(theta_idx, phi_idx);
    }
    
    // ========================================================================
    // VISUALIZATION - Get raw PDF value for a cell (for UI heatmap)
    // ========================================================================
    
    __host__ __device__ float getPDFValue(int theta_idx, int phi_idx) const {
        if (theta_idx < 0 || theta_idx >= GRID_RES || 
            phi_idx < 0 || phi_idx >= GRID_RES) return 0.0f;
        const int idx = theta_idx * GRID_RES + phi_idx;
        // For visualization, prefer precomp but fall back to local
        if (use_precomp && precomp_ptr) return precomp_ptr->pdf[idx];
        return pdf[idx];
    }
    
    __host__ __device__ float getMaxPDFValue() const {
        float max_val = 0.0f;
        for (int i = 0; i < GRID_SIZE; i++) {
            max_val = fmaxf(max_val, pdf[i]);
        }
        return max_val;
    }
    
    __host__ __device__ float getTotalWeight() const { return total_weight; }
    __host__ __device__ bool isValid() const { return is_valid; }
    
private:
    // ========================================================================
    // OPTIMIZED HELPER FUNCTIONS FOR 16x16 GRID
    // ========================================================================
    
    // Linear search optimized for small arrays (N <= 16) - faster than binary search!
    __device__ __forceinline__ int linearSearchCDF(const float* cdf, int size, float xi) const {
        xi = fminf(fmaxf(xi, 0.0f), 0.999999f);
        #pragma unroll
        for (int i = 0; i < size; i++) {
            if (xi < cdf[i]) return i;
        }
        return size - 1;
    }
    
    // Fast PDF computation for a cell using precomputed constants
    __device__ __forceinline__ float computePDFForCell(int theta_idx, int phi_idx) const {
        const int idx = theta_idx * GRID_RES + phi_idx;
        
        // Get PDF value from appropriate source
        float cell_value = use_precomp ? precomp_ptr->pdf[idx] : pdf[idx];
        if (cell_value < 1e-8f) return 1e-6f;
        
        float cell_prob = cell_value / fmaxf(total_weight, 1e-6f);
        
        // Solid angle using precomputed step sizes
        float theta_center = (theta_idx + 0.5f) * GRID_INV_HALF_RES * (M_PI * 0.5f);
        float sin_theta = fmaxf(sinf(theta_center), 0.01f);
        float solid_angle = sin_theta * GRID_D_THETA * GRID_D_PHI;
        
        return cell_prob / fmaxf(solid_angle, 1e-6f);
    }
    
    // Cosine-weighted PDF
    __device__ __forceinline__ float cosinePDF(const Vector3f& dir, const Vector3f& normal) const {
        return fmaxf(dot(dir, normal), 0.0f) * (1.0f / M_PI);
    }
    
    // Cosine-weighted hemisphere sampling (fallback)
    __device__ Vector3f sampleCosineHemisphere(const Vector3f& normal, curandState* rng, float& out_pdf) const {
        float u = curand_uniform(rng);
        float v = curand_uniform(rng);
        
        float r = sqrtf(u);
        float phi = 2.0f * M_PI * v;
        float sin_p, cos_p;
        sincosf(phi, &sin_p, &cos_p);
        float x = r * cos_p;
        float y = r * sin_p;
        float z = sqrtf(fmaxf(0.0f, 1.0f - u));
        
        Vector3f tangent, bitangent;
        buildFrame(normal, tangent, bitangent);
        
        Vector3f world_dir = unit_vector(tangent * x + bitangent * y + normal * z);
        out_pdf = fmaxf(dot(world_dir, normal), 0.0f) * (1.0f / M_PI);
        return world_dir;
    }
    
    // Build orthonormal frame (Frisvad's method)
    __device__ __forceinline__ void buildFrame(const Vector3f& n, Vector3f& t, Vector3f& b) const {
        if (n.z() < -0.9999999f) {
            t = Vector3f(0.0f, -1.0f, 0.0f);
            b = Vector3f(-1.0f, 0.0f, 0.0f);
            return;
        }
        const float a = 1.0f / (1.0f + n.z());
        const float c = -n.x() * n.y() * a;
        t = Vector3f(1.0f - n.x() * n.x() * a, c, -n.x());
        b = Vector3f(c, 1.0f - n.y() * n.y() * a, -n.y());
    }
    
    // World direction to spherical
    __device__ void worldToSpherical(const Vector3f& dir, const Vector3f& normal,
                                      float& theta, float& phi) const {
        Vector3f tangent, bitangent;
        buildFrame(normal, tangent, bitangent);
        
        float lx = dot(dir, tangent);
        float ly = dot(dir, bitangent);
        float lz = dot(dir, normal);
        
        theta = acosf(fminf(fmaxf(lz, -1.0f), 1.0f));
        phi = atan2f(ly, lx);
        if (phi < 0.0f) phi += 2.0f * M_PI;
    }
};

#endif // GRID_H
