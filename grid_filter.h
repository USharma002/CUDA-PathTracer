/**
 * @file grid_filter.h
 * @brief Bilateral filtering for radiosity grids
 * 
 * This file contains GPU kernels for smoothing radiosity grids using bilateral filtering.
 * Bilateral filtering preserves edges while smoothing noise, which is ideal for radiosity grids
 * where we want to smooth out noise but preserve directional importance.
 * 
 * To disable filtering: Set ENABLE_GRID_FILTERING to 0 or don't call the filter functions.
 * To remove completely: Delete this file and remove #include "grid_filter.h" from form_factors.h
 */

#ifndef GRID_FILTER_H
#define GRID_FILTER_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "vector.h"
#include "render_config.h"

// ============================================================================
// CONFIGURATION - Set to 0 to disable filtering
// ============================================================================
#define ENABLE_GRID_FILTERING 1

// Filter parameters
#define BILATERAL_KERNEL_RADIUS 2      // Spatial kernel radius (2 = 5x5 kernel)
#define BILATERAL_SIGMA_SPATIAL 1.5f   // Spatial sigma (controls spatial falloff)
#define BILATERAL_SIGMA_RANGE 0.3f     // Range sigma (controls intensity edge preservation)

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

__device__ inline float gaussianWeight(float distance, float sigma) {
    return expf(-(distance * distance) / (2.0f * sigma * sigma));
}

__device__ inline float luminanceFromRGB(const Vector3f& rgb) {
    return 0.2126f * rgb.x() + 0.7152f * rgb.y() + 0.0722f * rgb.z();
}

// ============================================================================
// BILATERAL FILTER FOR RADIOSITY GRID (Vector3f)
// ============================================================================

/**
 * @brief Applies bilateral filter to a single cell of the radiosity grid
 * 
 * Bilateral filtering: w(i,j) = exp(-|p-q|²/2σ_s²) * exp(-|I(p)-I(q)|²/2σ_r²)
 * - Spatial weight: closer pixels have more influence
 * - Range weight: similar intensities have more influence (preserves edges)
 */
__device__ Vector3f bilateralFilterCell(
    const Vector3f* input_grid,
    int center_i, int center_j,
    int grid_res,
    float sigma_spatial,
    float sigma_range
) {
    Vector3f center_val = input_grid[center_i * grid_res + center_j];
    float center_lum = luminanceFromRGB(center_val);
    
    Vector3f weighted_sum(0.0f, 0.0f, 0.0f);
    float total_weight = 0.0f;
    
    // Iterate over kernel neighborhood
    for (int di = -BILATERAL_KERNEL_RADIUS; di <= BILATERAL_KERNEL_RADIUS; di++) {
        for (int dj = -BILATERAL_KERNEL_RADIUS; dj <= BILATERAL_KERNEL_RADIUS; dj++) {
            // Handle boundary with wrapping (for phi) and clamping (for theta)
            int ni = center_i + di;
            int nj = (center_j + dj + grid_res) % grid_res;  // Wrap phi (azimuthal)
            
            // Clamp theta (polar) - don't wrap
            if (ni < 0 || ni >= grid_res) continue;
            
            Vector3f neighbor_val = input_grid[ni * grid_res + nj];
            float neighbor_lum = luminanceFromRGB(neighbor_val);
            
            // Spatial distance
            float spatial_dist = sqrtf((float)(di * di + dj * dj));
            float spatial_weight = gaussianWeight(spatial_dist, sigma_spatial);
            
            // Range (intensity) distance
            float range_dist = fabsf(center_lum - neighbor_lum);
            float range_weight = gaussianWeight(range_dist, sigma_range);
            
            // Combined bilateral weight
            float weight = spatial_weight * range_weight;
            
            weighted_sum += neighbor_val * weight;
            total_weight += weight;
        }
    }
    
    // Normalize
    if (total_weight > 1e-6f) {
        return weighted_sum / total_weight;
    }
    return center_val;
}

// ============================================================================
// CUDA KERNELS
// ============================================================================

/**
 * @brief CUDA kernel to apply bilateral filter to all radiosity grids
 * 
 * Grid layout: blockIdx.x = primitive index, blockIdx.y * blockDim + threadIdx = cell index
 */
__global__ void bilateral_filter_radiosity_grids_kernel(
    Primitive* primitives,
    int num_primitives,
    Vector3f* temp_buffer,  // Temporary buffer for filtered output
    int grid_res,
    float sigma_spatial,
    float sigma_range
) {
#if ENABLE_GRID_FILTERING
    int prim_idx = blockIdx.x;
    int cell_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (prim_idx >= num_primitives) return;
    if (cell_idx >= grid_res * grid_res) return;
    
    int i = cell_idx / grid_res;  // theta index
    int j = cell_idx % grid_res;  // phi index
    
    Vector3f* rad_grid = primitives[prim_idx].getRadiosityGrid();
    if (rad_grid == nullptr) return;
    
    // Calculate offset into temp buffer for this primitive
    int buffer_offset = prim_idx * grid_res * grid_res;
    
    // Apply bilateral filter
    temp_buffer[buffer_offset + cell_idx] = bilateralFilterCell(
        rad_grid, i, j, grid_res, sigma_spatial, sigma_range
    );
#endif
}

/**
 * @brief Copy filtered results back to radiosity grids
 */
__global__ void copy_filtered_to_grids_kernel(
    Primitive* primitives,
    int num_primitives,
    const Vector3f* temp_buffer,
    int grid_res
) {
#if ENABLE_GRID_FILTERING
    int prim_idx = blockIdx.x;
    int cell_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (prim_idx >= num_primitives) return;
    if (cell_idx >= grid_res * grid_res) return;
    
    Vector3f* rad_grid = primitives[prim_idx].getRadiosityGrid();
    if (rad_grid == nullptr) return;
    
    int buffer_offset = prim_idx * grid_res * grid_res;
    rad_grid[cell_idx] = temp_buffer[buffer_offset + cell_idx];
#endif
}

// ============================================================================
// HOST INTERFACE
// ============================================================================

/**
 * @brief Apply bilateral filtering to all radiosity grids
 * 
 * @param primitives Device pointer to primitives array
 * @param num_primitives Number of primitives
 * @param grid_res Grid resolution (GRID_RES)
 * @param sigma_spatial Spatial sigma for bilateral filter (default: BILATERAL_SIGMA_SPATIAL)
 * @param sigma_range Range sigma for bilateral filter (default: BILATERAL_SIGMA_RANGE)
 * 
 * Usage:
 *   filter_radiosity_grids(d_primitives, num_prims, GRID_RES);
 */
inline void filter_radiosity_grids(
    Primitive* d_primitives,
    int num_primitives,
    int grid_res,
    float sigma_spatial = BILATERAL_SIGMA_SPATIAL,
    float sigma_range = BILATERAL_SIGMA_RANGE
) {
#if ENABLE_GRID_FILTERING
    if (num_primitives <= 0) return;
    
    // Allocate temporary buffer for filtered output
    int total_cells = num_primitives * grid_res * grid_res;
    Vector3f* d_temp_buffer;
    cudaMalloc(&d_temp_buffer, total_cells * sizeof(Vector3f));
    
    // Launch filter kernel
    // Grid: (num_primitives, ceil(grid_cells / threads_per_block))
    // Block: threads_per_block threads
    int threads_per_block = 256;
    int cells_per_grid = grid_res * grid_res;  // e.g., 50*50 = 2500
    int blocks_for_cells = (cells_per_grid + threads_per_block - 1) / threads_per_block;
    
    dim3 grid_dim(num_primitives, blocks_for_cells);
    dim3 block_dim(threads_per_block);
    
    bilateral_filter_radiosity_grids_kernel<<<grid_dim, block_dim>>>(
        d_primitives, num_primitives, d_temp_buffer, grid_res,
        sigma_spatial, sigma_range
    );
    cudaDeviceSynchronize();
    
    // Copy filtered results back
    copy_filtered_to_grids_kernel<<<grid_dim, block_dim>>>(
        d_primitives, num_primitives, d_temp_buffer, grid_res
    );
    cudaDeviceSynchronize();
    
    // Free temporary buffer
    cudaFree(d_temp_buffer);
#endif
}

/**
 * @brief Apply bilateral filtering with default parameters
 */
inline void filter_radiosity_grids_default(Primitive* d_primitives, int num_primitives) {
    filter_radiosity_grids(d_primitives, num_primitives, GRID_RES);
}

// ============================================================================
// SIMPLE GAUSSIAN FILTER (Alternative, faster but doesn't preserve edges)
// ============================================================================

__device__ Vector3f gaussianFilterCell(
    const Vector3f* input_grid,
    int center_i, int center_j,
    int grid_res,
    float sigma
) {
    Vector3f weighted_sum(0.0f, 0.0f, 0.0f);
    float total_weight = 0.0f;
    
    for (int di = -BILATERAL_KERNEL_RADIUS; di <= BILATERAL_KERNEL_RADIUS; di++) {
        for (int dj = -BILATERAL_KERNEL_RADIUS; dj <= BILATERAL_KERNEL_RADIUS; dj++) {
            int ni = center_i + di;
            int nj = (center_j + dj + grid_res) % grid_res;
            
            if (ni < 0 || ni >= grid_res) continue;
            
            float dist = sqrtf((float)(di * di + dj * dj));
            float weight = gaussianWeight(dist, sigma);
            
            weighted_sum += input_grid[ni * grid_res + nj] * weight;
            total_weight += weight;
        }
    }
    
    if (total_weight > 1e-6f) {
        return weighted_sum / total_weight;
    }
    return input_grid[center_i * grid_res + center_j];
}

__global__ void gaussian_filter_radiosity_grids_kernel(
    Primitive* primitives,
    int num_primitives,
    Vector3f* temp_buffer,
    int grid_res,
    float sigma
) {
#if ENABLE_GRID_FILTERING
    int prim_idx = blockIdx.x;
    int cell_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (prim_idx >= num_primitives) return;
    if (cell_idx >= grid_res * grid_res) return;
    
    int i = cell_idx / grid_res;
    int j = cell_idx % grid_res;
    
    Vector3f* rad_grid = primitives[prim_idx].getRadiosityGrid();
    if (rad_grid == nullptr) return;
    
    int buffer_offset = prim_idx * grid_res * grid_res;
    temp_buffer[buffer_offset + cell_idx] = gaussianFilterCell(rad_grid, i, j, grid_res, sigma);
#endif
}

/**
 * @brief Apply simple Gaussian filtering (faster, but blurs edges)
 */
inline void filter_radiosity_grids_gaussian(
    Primitive* d_primitives,
    int num_primitives,
    int grid_res,
    float sigma = BILATERAL_SIGMA_SPATIAL
) {
#if ENABLE_GRID_FILTERING
    if (num_primitives <= 0) return;
    
    int total_cells = num_primitives * grid_res * grid_res;
    Vector3f* d_temp_buffer;
    cudaMalloc(&d_temp_buffer, total_cells * sizeof(Vector3f));
    
    int threads_per_block = 256;
    int cells_per_grid = grid_res * grid_res;
    int blocks_for_cells = (cells_per_grid + threads_per_block - 1) / threads_per_block;
    
    dim3 grid_dim(num_primitives, blocks_for_cells);
    dim3 block_dim(threads_per_block);
    
    gaussian_filter_radiosity_grids_kernel<<<grid_dim, block_dim>>>(
        d_primitives, num_primitives, d_temp_buffer, grid_res, sigma
    );
    cudaDeviceSynchronize();
    
    copy_filtered_to_grids_kernel<<<grid_dim, block_dim>>>(
        d_primitives, num_primitives, d_temp_buffer, grid_res
    );
    cudaDeviceSynchronize();
    
    cudaFree(d_temp_buffer);
#endif
}

#endif // GRID_FILTER_H
