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
#include "core/vector.h"
#include "rendering/render_config.h"

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

// ============================================================================
// FLOAT GRID FILTERS & PDF PREPROCESSING (Form-factor floats + Radiosity luminance)
// ============================================================================

#if ENABLE_GRID_FILTERING

/**
 * Compute luminance (Y) from radiosity Vector3f grids into a contiguous float buffer.
 * Output layout: out_lums[prim_idx * grid_res*grid_res + cell_idx]
 */
__global__ void compute_radiosity_luminance_kernel(Primitive* primitives, float* out_lums, int num_primitives, int grid_res) {
    int prim_idx = blockIdx.x;
    int cell_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (prim_idx >= num_primitives) return;
    if (cell_idx >= grid_res * grid_res) return;

    Vector3f* rad_grid = primitives[prim_idx].getRadiosityGrid();
    if (rad_grid == nullptr) return;

    Vector3f v = rad_grid[cell_idx];
    float lum = luminanceFromRGB(v);
    out_lums[prim_idx * grid_res * grid_res + cell_idx] = lum;
}

/** Copy form-factor float grids (inline in Primitive) into contiguous device buffer */
__global__ void copy_formfactor_to_buffer_kernel(Primitive* primitives, float* out_buf, int num_primitives, int grid_res) {
    int prim_idx = blockIdx.x;
    int cell_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (prim_idx >= num_primitives) return;
    if (cell_idx >= grid_res * grid_res) return;

    float val = 0.0f;
    // Access primitive grid
    if (primitives[prim_idx].type == PRIM_TRIANGLE) {
        val = primitives[prim_idx].tri.grid[cell_idx];
    } else {
        val = primitives[prim_idx].quad.grid[cell_idx];
    }
    out_buf[prim_idx * grid_res * grid_res + cell_idx] = val;
}

__device__ inline float gaussianFilterCellFloat(const float* input_grid, int center_i, int center_j, int grid_res, float sigma) {
    float weighted_sum = 0.0f;
    float total_weight = 0.0f;
    for (int di = -BILATERAL_KERNEL_RADIUS; di <= BILATERAL_KERNEL_RADIUS; di++) {
        for (int dj = -BILATERAL_KERNEL_RADIUS; dj <= BILATERAL_KERNEL_RADIUS; dj++) {
            int ni = center_i + di;
            int nj = (center_j + dj + grid_res) % grid_res;
            if (ni < 0 || ni >= grid_res) continue;
            float dist = sqrtf((float)(di*di + dj*dj));
            float w = gaussianWeight(dist, sigma);
            weighted_sum += input_grid[ni * grid_res + nj] * w;
            total_weight += w;
        }
    }
    if (total_weight > 1e-6f) return weighted_sum / total_weight;
    return input_grid[center_i * grid_res + center_j];
}

__global__ void gaussian_filter_float_kernel(const float* input_buf, float* output_buf, int num_primitives, int grid_res, float sigma) {
    int prim_idx = blockIdx.x;
    int cell_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (prim_idx >= num_primitives) return;
    if (cell_idx >= grid_res * grid_res) return;
    int i = cell_idx / grid_res;
    int j = cell_idx % grid_res;
    const float* src = input_buf + prim_idx * grid_res * grid_res;
    float* dst = output_buf + prim_idx * grid_res * grid_res;
    dst[cell_idx] = gaussianFilterCellFloat(src, i, j, grid_res, sigma);
}

__global__ void bilateral_filter_float_kernel(const float* input_buf, float* output_buf, int num_primitives, int grid_res, float sigma_spatial, float sigma_range) {
    int prim_idx = blockIdx.x;
    int cell_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (prim_idx >= num_primitives) return;
    if (cell_idx >= grid_res * grid_res) return;
    int ci = cell_idx / grid_res;
    int cj = cell_idx % grid_res;

    const float* src = input_buf + prim_idx * grid_res * grid_res;
    float center = src[cell_idx];

    float weighted_sum = 0.0f;
    float total_weight = 0.0f;
    for (int di = -BILATERAL_KERNEL_RADIUS; di <= BILATERAL_KERNEL_RADIUS; di++) {
        for (int dj = -BILATERAL_KERNEL_RADIUS; dj <= BILATERAL_KERNEL_RADIUS; dj++) {
            int ni = ci + di;
            int nj = (cj + dj + grid_res) % grid_res;
            if (ni < 0 || ni >= grid_res) continue;
            float spatial = gaussianWeight(sqrtf((float)(di*di + dj*dj)), sigma_spatial);
            float range = gaussianWeight(fabsf(center - src[ni * grid_res + nj]), sigma_range);
            float w = spatial * range;
            weighted_sum += src[ni * grid_res + nj] * w;
            total_weight += w;
        }
    }
    if (total_weight > 1e-6f) output_buf[prim_idx * grid_res * grid_res + cell_idx] = weighted_sum / total_weight;
    else output_buf[prim_idx * grid_res * grid_res + cell_idx] = center;
}

// Normalize per-primitive so each primitive's PDF sums to 1 (or leave zeros as-is)
__global__ void normalize_pdf_kernel(float* buf, int num_primitives, int grid_res) {
    int prim_idx = blockIdx.x;
    if (prim_idx >= num_primitives) return;
    int cells = grid_res * grid_res;
    float sum = 0.0f;
    float* base = buf + prim_idx * cells;
    for (int i = 0; i < cells; i++) sum += base[i];
    if (sum <= 1e-12f) return;
    for (int i = 0; i < cells; i++) base[i] = base[i] / sum;
}

/**
 * Host wrapper: prepare and optionally filter PDFs for all primitives.
 * - Copies formfactor float grids into `out_formfactor` and computes radiosity luminance into `out_radiosity`.
 * - Applies either bilateral or gaussian filter to each buffer and normalizes per-primitive.
 */
inline void filter_pdfs_for_primitives(Primitive* d_primitives,
                                      float* d_out_formfactor,
                                      float* d_out_radiosity,
                                      int num_primitives,
                                      int grid_res,
                                      bool use_bilateral = true,
                                      float sigma_spatial = BILATERAL_SIGMA_SPATIAL,
                                      float sigma_range = BILATERAL_SIGMA_RANGE) {
    if (!d_primitives || num_primitives <= 0) return;

    int threads_per_block = 256;
    int cells_per_grid = grid_res * grid_res;
    int blocks_for_cells = (cells_per_grid + threads_per_block - 1) / threads_per_block;
    dim3 grid_dim(num_primitives, blocks_for_cells);
    dim3 block_dim(threads_per_block);

    // 1) Copy formfactor floats into d_out_formfactor
    copy_formfactor_to_buffer_kernel<<<grid_dim, block_dim>>>(d_primitives, d_out_formfactor, num_primitives, grid_res);
    cudaDeviceSynchronize();

    // 2) Compute luminance from radiosity into d_out_radiosity
    compute_radiosity_luminance_kernel<<<grid_dim, block_dim>>>(d_primitives, d_out_radiosity, num_primitives, grid_res);
    cudaDeviceSynchronize();

    // Allocate temp buffer
    size_t total_cells = (size_t)num_primitives * cells_per_grid;
    float* d_temp;
    cudaMalloc(&d_temp, total_cells * sizeof(float));

    // 3) Filter both buffers
    if (use_bilateral) {
        bilateral_filter_float_kernel<<<grid_dim, block_dim>>>(d_out_formfactor, d_temp, num_primitives, grid_res, sigma_spatial, sigma_range);
        cudaDeviceSynchronize();
        // swap results back into out buffer
        cudaMemcpy(d_out_formfactor, d_temp, total_cells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        bilateral_filter_float_kernel<<<grid_dim, block_dim>>>(d_out_radiosity, d_temp, num_primitives, grid_res, sigma_spatial, sigma_range);
        cudaDeviceSynchronize();
        cudaMemcpy(d_out_radiosity, d_temp, total_cells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    } else {
        gaussian_filter_float_kernel<<<grid_dim, block_dim>>>(d_out_formfactor, d_temp, num_primitives, grid_res, sigma_spatial);
        cudaDeviceSynchronize();
        cudaMemcpy(d_out_formfactor, d_temp, total_cells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        gaussian_filter_float_kernel<<<grid_dim, block_dim>>>(d_out_radiosity, d_temp, num_primitives, grid_res, sigma_spatial);
        cudaDeviceSynchronize();
        cudaMemcpy(d_out_radiosity, d_temp, total_cells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }

    // 4) Normalize per-primitive
    normalize_pdf_kernel<<<num_primitives, 1>>>(d_out_formfactor, num_primitives, grid_res);
    cudaDeviceSynchronize();
    normalize_pdf_kernel<<<num_primitives, 1>>>(d_out_radiosity, num_primitives, grid_res);
    cudaDeviceSynchronize();

    cudaFree(d_temp);
}

#endif // ENABLE_GRID_FILTERING

