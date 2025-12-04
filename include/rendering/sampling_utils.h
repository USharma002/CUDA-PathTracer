/**
 * @file sampling_utils.h
 * @brief GPU sampling utilities for Monte Carlo path tracing with radiosity-guided importance sampling
 * 
 * Implements 2D inverse CDF sampling from directional grids (form factors, radiosity) using
 * spherical coordinate parameterization. Includes MIS (Multiple Importance Sampling) with
 * balance heuristic combining grid-based and BSDF sampling.
 */

#ifndef SAMPLING_UTILS_H
#define SAMPLING_UTILS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "core/vector.h"
#include "rendering/render_config.h"

// ============================================================================
// GRID INDEXING - Row-major layout: grid[row * cols + col]
// ============================================================================

__device__ inline float getGridValueFast(const float* grid, int i, int j, int grid_res) {
    return grid[i * grid_res + j];
}

__device__ inline int gridCoordTo1D(int i, int j, int grid_res) {
    return i * grid_res + j;
}

__device__ inline void gridCoordFrom1D(int idx, int& i, int& j, int grid_res) {
    i = idx / grid_res;
    j = idx % grid_res;
}

// ============================================================================
// 2D INVERSE CDF SAMPLING - Hierarchical: sample row (marginal), then column (conditional)
// ============================================================================

// Build marginal CDF over rows for importance sampling
__device__ void buildGridPDFFast(const float* grid, int grid_res, float* cdf_y, float& total_weight) {
    float row_sums[DEFAULT_GRID_RES];
    total_weight = 0.0f;
    
    #pragma unroll 4
    for (int v = 0; v < grid_res; v++) {
        float row_sum = 0.0f;
        #pragma unroll 4
        for (int u = 0; u < grid_res; u++)
            row_sum += getGridValueFast(grid, v, u, grid_res);
        row_sums[v] = row_sum;
        total_weight += row_sum;
    }
    
    float running = 0.0f;
    for (int v = 0; v < grid_res; v++) {
        running += row_sums[v];
        cdf_y[v] = running / fmaxf(total_weight, 1e-6f);
    }
}

// Binary search: find i where cdf[i] <= xi < cdf[i+1]
__device__ inline int binarySearchCDFFast(const float* cdf, int size, float xi) {
    xi = fminf(fmaxf(xi, 0.0f), 0.999999f);
    int left = 0, right = size - 1;
    while (left < right) {
        int mid = (left + right) >> 1;
        if (xi < cdf[mid + 1]) right = mid;
        else left = mid + 1;
    }
    return left;
}

// Build conditional CDF for single row (on-demand to minimize memory)
__device__ void buildConditionalCDFFast(const float* grid, int v_idx, int grid_res, float* cdf_x) {
    float row_sum = 0.0f;
    for (int u = 0; u < grid_res; u++)
        row_sum += getGridValueFast(grid, v_idx, u, grid_res);
    
    if (row_sum < 1e-6f) {
        for (int u = 0; u < grid_res; u++) cdf_x[u] = (u + 1.0f) / grid_res;
        return;
    }
    
    float running = 0.0f;
    for (int u = 0; u < grid_res; u++) {
        running += getGridValueFast(grid, v_idx, u, grid_res);
        cdf_x[u] = running / row_sum;
    }
}

// ============================================================================
// LOCAL COORDINATE FRAME - Frisvad's robust method
// ============================================================================

__device__ void buildCoordinateSystemFast(const Vector3f &n, Vector3f &tangent, Vector3f &bitangent) {
    if (n.z() < -0.9999999f) {
        tangent = Vector3f(0.0f, -1.0f, 0.0f);
        bitangent = Vector3f(-1.0f, 0.0f, 0.0f);
        return;
    }
    float a = 1.0f / (1.0f + n.z());
    float b = -n.x() * n.y() * a;
    tangent = Vector3f(1.0f - n.x() * n.x() * a, b, -n.x());
    bitangent = Vector3f(b, 1.0f - n.y() * n.y() * a, -n.y());
}

__device__ Vector3f localToWorldFast(const Vector3f &local, const Vector3f &normal, 
                                      const Vector3f &tangent, const Vector3f &bitangent) {
    return tangent * local.x() + bitangent * local.y() + normal * local.z();
}

// ============================================================================
// COSINE-WEIGHTED HEMISPHERE SAMPLING - Malley's method, PDF = cos(θ)/π
// ============================================================================

__device__ Vector3f uvToDirectionFast(float u, float v) {
    float r = sqrtf(u);
    float theta = 2.0f * M_PI * v;
    return Vector3f(r * cosf(theta), r * sinf(theta), sqrtf(fmaxf(0.0f, 1.0f - u)));
}

__device__ Vector3f sampleCosineWeightedHemisphereFast(const Vector3f &normal, curandState *rand_state) {
    Vector3f local_dir = uvToDirectionFast(curand_uniform(rand_state), curand_uniform(rand_state));
    Vector3f tangent, bitangent;
    buildCoordinateSystemFast(normal, tangent, bitangent);
    return unit_vector(localToWorldFast(local_dir, normal, tangent, bitangent));
}

// ============================================================================
// LUMINANCE CONVERSION - ITU-R BT.709 (used for importance sampling weights)
// ============================================================================

__device__ inline float rgbToLuminance(const Vector3f& rgb) {
    return 0.2126f * rgb.x() + 0.7152f * rgb.y() + 0.0722f * rgb.z();
}

// ============================================================================
// RADIOSITY GRID SAMPLING - Sample directions proportional to incoming luminance
// Grid layout: row = θ (polar angle), col = φ (azimuthal angle)
// Uses luminance for importance sampling (matches reference implementation)
// ============================================================================

__device__ Vector3f sampleRadiosityGridFast(const Vector3f* radiosity_grid, int grid_res,
                                            const Vector3f &normal, curandState *rand_state,
                                            bool &success) {
    // Compute marginal distribution using luminance (row sums)
    float total_rad = 0.0f, row_sums[DEFAULT_GRID_RES];
    for (int v = 0; v < grid_res; v++) {
        float row_sum = 0.0f;
        for (int u = 0; u < grid_res; u++) {
            row_sum += rgbToLuminance(radiosity_grid[v * grid_res + u]);
        }
        row_sums[v] = row_sum;
        total_rad += row_sum;
    }
    
    if (total_rad < 1e-6f) { success = false; return Vector3f(0, 0, 0); }
    
    // Build and sample marginal CDF (θ dimension)
    float cdf_y[DEFAULT_GRID_RES], running = 0.0f;
    for (int v = 0; v < grid_res; v++) {
        running += row_sums[v];
        cdf_y[v] = running / total_rad;
    }
    int theta_idx = binarySearchCDFFast(cdf_y, grid_res, curand_uniform(rand_state));
    
    // Build and sample conditional CDF (φ|θ) using luminance
    float cdf_x[DEFAULT_GRID_RES];
    running = 0.0f;
    for (int u = 0; u < grid_res; u++) {
        running += rgbToLuminance(radiosity_grid[theta_idx * grid_res + u]);
        cdf_x[u] = running / fmaxf(row_sums[theta_idx], 1e-6f);
    }
    int phi_idx = binarySearchCDFFast(cdf_x, grid_res, curand_uniform(rand_state));
    
    // Grid indices → spherical coords with stratified jitter
    float theta = fminf(((theta_idx + curand_uniform(rand_state)) / grid_res) * M_PI, M_PI * 0.5f - 0.01f);
    float phi = ((phi_idx + curand_uniform(rand_state)) / grid_res) * 2.0f * M_PI;
    
    // Spherical → Cartesian → world space
    float sin_t = sinf(theta), cos_t = cosf(theta);
    Vector3f local_dir(sin_t * cosf(phi), sin_t * sinf(phi), cos_t);
    Vector3f tangent, bitangent;
    buildCoordinateSystemFast(normal, tangent, bitangent);
    
    success = true;
    return unit_vector(localToWorldFast(local_dir, normal, tangent, bitangent));
}

// ============================================================================
// TOP-K SAMPLING - Uniform selection from highest-weight cells
// ============================================================================

__device__ Vector3f sampleTopKHistogramFast(const int* top_k_indices, int top_k_count, 
                                            int grid_res, const Vector3f &normal,
                                            curandState *rand_state) {
    if (top_k_count <= 0) return sampleCosineWeightedHemisphereFast(normal, rand_state);
    
    int idx = fminf((int)(curand_uniform(rand_state) * top_k_count), top_k_count - 1);
    int theta_idx, phi_idx;
    gridCoordFrom1D(top_k_indices[idx], theta_idx, phi_idx, grid_res);
    
    float theta = fminf(((theta_idx + curand_uniform(rand_state)) / grid_res) * M_PI, M_PI * 0.5f - 0.01f);
    float phi = ((phi_idx + curand_uniform(rand_state)) / grid_res) * 2.0f * M_PI;
    
    float sin_t = sinf(theta), cos_t = cosf(theta);
    Vector3f local_dir(sin_t * cosf(phi), sin_t * sinf(phi), cos_t);
    Vector3f tangent, bitangent;
    buildCoordinateSystemFast(normal, tangent, bitangent);
    return unit_vector(localToWorldFast(local_dir, normal, tangent, bitangent));
}

// ============================================================================
// PDF COMPUTATION - For MIS weight calculation
// ============================================================================

// Power heuristic: w_a = p_a² / (p_a² + p_b²)
__device__ inline float misWeightPowerHeuristic(float pdf_a, float pdf_b) {
    if (pdf_a <= 0.0f) return 0.0f;
    float a2 = pdf_a * pdf_a, b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2);
}

// Cosine-weighted hemisphere: p(ω) = cos(θ)/π
__device__ inline float cosinePDF(const Vector3f& dir, const Vector3f& normal) {
    return fmaxf(dot(dir, normal), 0.0f) / M_PI;
}

// Grid sampling PDF: p(ω) = p(cell) / solid_angle(cell)
// Only considers upper hemisphere
__device__ inline float gridPDF(const float* grid, int grid_res, const Vector3f& dir, 
                                const Vector3f& normal, float total_weight) {
    // Compute hemisphere weight
    int half_res = grid_res / 2;
    float hemisphere_weight = 0.0f;
    for (int v = 0; v < half_res; v++) {
        for (int u = 0; u < grid_res; u++) {
            hemisphere_weight += getGridValueFast(grid, v, u, grid_res);
        }
    }
    if (hemisphere_weight <= 1e-6f) return 0.0f;
    
    // World → local coordinates
    Vector3f tangent, bitangent;
    buildCoordinateSystemFast(normal, tangent, bitangent);
    float lx = dot(dir, tangent), ly = dot(dir, bitangent), lz = dot(dir, normal);
    if (lz <= 0.0f) return 0.0f;  // Below hemisphere
    
    // Local → spherical (θ, φ)
    float r = sqrtf(lx*lx + ly*ly + lz*lz);
    float theta = (r > 0.0f) ? acosf(fminf(lz / r, 1.0f)) : 0.0f;
    float phi = atan2f(ly, lx);
    if (phi < 0.0f) phi += 2.0f * M_PI;
    
    // Map theta to grid index (hemisphere only: [0, π/2] -> [0, half_res-1])
    int gi = (int)fminf((theta / (M_PI * 0.5f)) * half_res, half_res - 1);
    int gj = (int)fminf((phi / (2.0f * M_PI)) * grid_res, grid_res - 1);
    gi = max(0, min(gi, half_res - 1));
    gj = max(0, min(gj, grid_res - 1));
    
    // PDF = discrete_probability / solid_angle
    float cell_prob = getGridValueFast(grid, gi, gj, grid_res) / hemisphere_weight;
    float sin_theta = fmaxf(sinf(theta), 0.01f);
    float d_theta = (M_PI * 0.5f) / half_res;
    float d_phi = 2.0f * M_PI / grid_res;
    float solid_angle = sin_theta * d_theta * d_phi;
    return cell_prob / fmaxf(solid_angle, 1e-6f);
}

// ============================================================================
// RADIOSITY GRID UTILITIES - Using Luminance for proper importance sampling
// Uses rgbToLuminance() defined earlier
// ============================================================================

__device__ inline float computeRadiosityTotalWeight(const Vector3f* grid, int grid_res) {
    // Only count upper hemisphere (first half of theta rows)
    int half_res = grid_res / 2;
    float total = 0.0f;
    for (int v = 0; v < half_res; v++) {
        for (int u = 0; u < grid_res; u++) {
            total += rgbToLuminance(grid[v * grid_res + u]);
        }
    }
    return total;
}

__device__ inline float getRadiosityGridValue(const Vector3f* grid, int i, int j, int grid_res) {
    Vector3f rad = grid[i * grid_res + j];
    return rgbToLuminance(rad);
}

__device__ inline float radiosityGridPDF(const Vector3f* radiosity_grid, int grid_res, 
                                          const Vector3f& dir, const Vector3f& normal, 
                                          float total_weight) {
    // Compute hemisphere weight (same as sampling uses)
    int half_res = grid_res / 2;
    float hemisphere_weight = 0.0f;
    for (int v = 0; v < half_res; v++) {
        for (int u = 0; u < grid_res; u++) {
            hemisphere_weight += getRadiosityGridValue(radiosity_grid, v, u, grid_res);
        }
    }
    if (hemisphere_weight <= 1e-6f) return 0.0f;
    
    Vector3f tangent, bitangent;
    buildCoordinateSystemFast(normal, tangent, bitangent);
    float lx = dot(dir, tangent), ly = dot(dir, bitangent), lz = dot(dir, normal);
    if (lz <= 0.0f) return 0.0f;  // Below hemisphere
    
    float r = sqrtf(lx*lx + ly*ly + lz*lz);
    float theta = (r > 0.0f) ? acosf(fminf(lz / r, 1.0f)) : 0.0f;
    float phi = atan2f(ly, lx);
    if (phi < 0.0f) phi += 2.0f * M_PI;
    
    // Map theta to grid index (hemisphere only: [0, π/2] -> [0, half_res-1])
    int gi = (int)fminf((theta / (M_PI * 0.5f)) * half_res, half_res - 1);
    int gj = (int)fminf((phi / (2.0f * M_PI)) * grid_res, grid_res - 1);
    gi = max(0, min(gi, half_res - 1));
    gj = max(0, min(gj, grid_res - 1));
    
    float cell_prob = getRadiosityGridValue(radiosity_grid, gi, gj, grid_res) / hemisphere_weight;
    float sin_theta = fmaxf(sinf(theta), 0.01f);
    float d_theta = (M_PI * 0.5f) / half_res;
    float d_phi = 2.0f * M_PI / grid_res;
    float solid_angle = sin_theta * d_theta * d_phi;
    return cell_prob / fmaxf(solid_angle, 1e-6f);
}

// ============================================================================
// GRID SAMPLING WITH PDF OUTPUT - For MIS weight computation
// ============================================================================

// Sample from radiosity grid, outputs direction and its PDF
// IMPORTANT: Grid covers full sphere but we only sample upper hemisphere
__device__ Vector3f sampleRadiosityGridSpherical(const Vector3f* radiosity_grid, int grid_res, 
                                                  const Vector3f& normal, curandState* rand_state,
                                                  float& pdf_out, float total_weight) {
    // Only consider upper hemisphere (first half of theta range)
    // Grid: theta in [0, π], we want [0, π/2]
    int half_res = grid_res / 2;  // Only sample from upper hemisphere rows
    
    float cdf_y[DEFAULT_GRID_RES], row_sums[DEFAULT_GRID_RES], running = 0.0f;
    float hemisphere_weight = 0.0f;
    
    // Build CDF only for upper hemisphere (theta < π/2)
    for (int v = 0; v < half_res; v++) {
        float row_sum = 0.0f;
        for (int u = 0; u < grid_res; u++)
            row_sum += getRadiosityGridValue(radiosity_grid, v, u, grid_res);
        row_sums[v] = row_sum;
        hemisphere_weight += row_sum;
        running += row_sum;
        cdf_y[v] = running;
    }
    
    // Normalize CDF
    if (hemisphere_weight < 1e-6f) {
        // Fallback to cosine sampling if no energy in upper hemisphere
        pdf_out = cosinePDF(sampleCosineWeightedHemisphereFast(normal, rand_state), normal);
        return sampleCosineWeightedHemisphereFast(normal, rand_state);
    }
    for (int v = 0; v < half_res; v++) {
        cdf_y[v] /= hemisphere_weight;
    }
    
    int theta_idx = binarySearchCDFFast(cdf_y, half_res, curand_uniform(rand_state));
    
    float cdf_x[DEFAULT_GRID_RES], row_sum = row_sums[theta_idx];
    if (row_sum < 1e-6f) {
        for (int u = 0; u < grid_res; u++) cdf_x[u] = (u + 1.0f) / grid_res;
    } else {
        float rs = 0.0f;
        for (int u = 0; u < grid_res; u++) {
            rs += getRadiosityGridValue(radiosity_grid, theta_idx, u, grid_res);
            cdf_x[u] = rs / row_sum;
        }
    }
    int phi_idx = binarySearchCDFFast(cdf_x, grid_res, curand_uniform(rand_state));
    
    // Convert to spherical - theta now in [0, π/2] (upper hemisphere only)
    float theta = ((theta_idx + curand_uniform(rand_state)) / half_res) * (M_PI * 0.5f);
    theta = fminf(theta, M_PI * 0.5f - 0.01f);  // Stay slightly away from horizon
    float phi = ((phi_idx + curand_uniform(rand_state)) / grid_res) * 2.0f * M_PI;
    
    float sin_t = sinf(theta), cos_t = cosf(theta);
    Vector3f local_dir(sin_t * cosf(phi), sin_t * sinf(phi), cos_t);
    Vector3f tangent, bitangent;
    buildCoordinateSystemFast(normal, tangent, bitangent);
    Vector3f world_dir = unit_vector(localToWorldFast(local_dir, normal, tangent, bitangent));
    
    // PDF: probability of this cell / solid angle of cell
    float cell_prob = getRadiosityGridValue(radiosity_grid, theta_idx, phi_idx, grid_res) / fmaxf(hemisphere_weight, 1e-6f);
    float d_theta = (M_PI * 0.5f) / half_res;  // Adjusted for hemisphere
    float d_phi = 2.0f * M_PI / grid_res;
    float solid_angle = fmaxf(sin_t, 0.01f) * d_theta * d_phi;
    pdf_out = cell_prob / fmaxf(solid_angle, 1e-6f);
    
    return world_dir;
}

// Sample from form factor grid, outputs direction and its PDF
// Only samples from upper hemisphere
__device__ Vector3f sampleGridDirectionSpherical(const float* grid, int grid_res, 
                                                  const Vector3f& normal, curandState* rand_state,
                                                  float& pdf_out, float total_weight) {
    int half_res = grid_res / 2;
    float cdf_y[DEFAULT_GRID_RES], row_sums[DEFAULT_GRID_RES], running = 0.0f;
    float hemisphere_weight = 0.0f;
    
    // Build CDF only for upper hemisphere
    for (int v = 0; v < half_res; v++) {
        float row_sum = 0.0f;
        for (int u = 0; u < grid_res; u++)
            row_sum += getGridValueFast(grid, v, u, grid_res);
        row_sums[v] = row_sum;
        hemisphere_weight += row_sum;
        running += row_sum;
        cdf_y[v] = running;
    }
    
    if (hemisphere_weight < 1e-6f) {
        Vector3f dir = sampleCosineWeightedHemisphereFast(normal, rand_state);
        pdf_out = cosinePDF(dir, normal);
        return dir;
    }
    for (int v = 0; v < half_res; v++) {
        cdf_y[v] /= hemisphere_weight;
    }
    
    int theta_idx = binarySearchCDFFast(cdf_y, half_res, curand_uniform(rand_state));
    
    float cdf_x[DEFAULT_GRID_RES], row_sum = row_sums[theta_idx];
    if (row_sum < 1e-6f) {
        for (int u = 0; u < grid_res; u++) cdf_x[u] = (u + 1.0f) / grid_res;
    } else {
        float rs = 0.0f;
        for (int u = 0; u < grid_res; u++) {
            rs += getGridValueFast(grid, theta_idx, u, grid_res);
            cdf_x[u] = rs / row_sum;
        }
    }
    int phi_idx = binarySearchCDFFast(cdf_x, grid_res, curand_uniform(rand_state));
    
    // Convert to spherical - theta in [0, π/2]
    float theta = ((theta_idx + curand_uniform(rand_state)) / half_res) * (M_PI * 0.5f);
    theta = fminf(theta, M_PI * 0.5f - 0.01f);
    float phi = ((phi_idx + curand_uniform(rand_state)) / grid_res) * 2.0f * M_PI;
    
    float sin_t = sinf(theta), cos_t = cosf(theta);
    Vector3f local_dir(sin_t * cosf(phi), sin_t * sinf(phi), cos_t);
    Vector3f tangent, bitangent;
    buildCoordinateSystemFast(normal, tangent, bitangent);
    Vector3f world_dir = unit_vector(localToWorldFast(local_dir, normal, tangent, bitangent));
    
    float cell_prob = getGridValueFast(grid, theta_idx, phi_idx, grid_res) / fmaxf(hemisphere_weight, 1e-6f);
    float d_theta = (M_PI * 0.5f) / half_res;
    float d_phi = 2.0f * M_PI / grid_res;
    float solid_angle = fmaxf(sin_t, 0.01f) * d_theta * d_phi;
    pdf_out = cell_prob / fmaxf(solid_angle, 1e-6f);
    
    return world_dir;
}

// ============================================================================
// MULTIPLE IMPORTANCE SAMPLING (MIS) - One-sample balance heuristic
// 
// Sample from either BSDF or guided distribution, then apply proper MIS weight.
// ============================================================================

// Configurable BSDF sampling fraction (0.5 = equal mix)
#define BSDF_SAMPLING_FRACTION 0.5f

__device__ Vector3f sampleMISFastRadiosity(const Vector3f* radiosity_grid, int grid_res, 
                                           const Vector3f &normal, curandState *rand_state, 
                                           float &weight) {
    float total_weight = computeRadiosityTotalWeight(radiosity_grid, grid_res);
    
    // If no meaningful guiding data, fall back to pure BSDF sampling
    if (total_weight < 1e-4f) {
        weight = 1.0f;
        return sampleCosineWeightedHemisphereFast(normal, rand_state);
    }
    
    // Decide whether to sample from BSDF or guided distribution
    float xi = curand_uniform(rand_state);
    bool use_bsdf = (xi < BSDF_SAMPLING_FRACTION);
    Vector3f dir;
    float pdf_guided, pdf_bsdf;
    
    if (use_bsdf) {
        // Sample from BSDF (cosine-weighted hemisphere)
        dir = sampleCosineWeightedHemisphereFast(normal, rand_state);
        pdf_bsdf = cosinePDF(dir, normal);
        pdf_guided = radiosityGridPDF(radiosity_grid, grid_res, dir, normal, total_weight);
    } else {
        // Sample from guided distribution (radiosity grid)
        dir = sampleRadiosityGridSpherical(radiosity_grid, grid_res, normal, rand_state, pdf_guided, total_weight);
        pdf_bsdf = cosinePDF(dir, normal);
    }
    
    // If guided PDF is negligible, treat as pure BSDF sampling
    if (pdf_guided < 1e-6f) {
        weight = 1.0f;
        return dir;
    }
    
    // Balance heuristic: mixed_pdf = α * pdf_bsdf + (1-α) * pdf_guided
    float mixed_pdf = BSDF_SAMPLING_FRACTION * pdf_bsdf + (1.0f - BSDF_SAMPLING_FRACTION) * pdf_guided;
    
    // For Lambertian: bsdf = albedo/π, integrator does throughput *= albedo
    // Full contribution: (albedo/π) * cos(θ) / mixed_pdf
    // Since integrator does *= albedo, weight = cos(θ) / (π * mixed_pdf)
    float cos_theta = fmaxf(dot(dir, normal), 0.0f);
    
    if (mixed_pdf > 1e-6f && cos_theta > 0.0f) {
        weight = cos_theta / (M_PI * mixed_pdf);
        // Clamp to reasonable range to prevent fireflies
        weight = fmaxf(0.01f, fminf(weight, 4.0f));
    } else {
        weight = 0.0f;
    }
    
    return dir;
}

// MIS with form factor grid
__device__ Vector3f sampleMISFast(const float* grid, int grid_res, const Vector3f &normal, 
                                  curandState *rand_state, float &weight) {
    float total_weight = 0.0f;
    for (int i = 0; i < grid_res * grid_res; i++) total_weight += grid[i];
    
    if (total_weight < 1e-4f) {
        weight = 1.0f;  // Fall back to BSDF sampling
        return sampleCosineWeightedHemisphereFast(normal, rand_state);
    }
    
    // Decide whether to sample from BSDF or guided distribution
    float xi = curand_uniform(rand_state);
    bool use_bsdf = (xi < BSDF_SAMPLING_FRACTION);
    Vector3f dir;
    float pdf_guided, pdf_bsdf;
    
    if (use_bsdf) {
        // Sample from BSDF (cosine-weighted hemisphere)
        dir = sampleCosineWeightedHemisphereFast(normal, rand_state);
        pdf_bsdf = cosinePDF(dir, normal);
        pdf_guided = gridPDF(grid, grid_res, dir, normal, total_weight);
    } else {
        // Sample from guided distribution (form factor grid)
        dir = sampleGridDirectionSpherical(grid, grid_res, normal, rand_state, pdf_guided, total_weight);
        pdf_bsdf = cosinePDF(dir, normal);
    }
    
    // If guided PDF is negligible, treat as pure BSDF sampling
    if (pdf_guided < 1e-6f) {
        weight = 1.0f;
        return dir;
    }
    
    // Balance heuristic mixed PDF
    float mixed_pdf = BSDF_SAMPLING_FRACTION * pdf_bsdf + (1.0f - BSDF_SAMPLING_FRACTION) * pdf_guided;
    
    // Weight = cos_theta / (π * mixed_pdf)
    float cos_theta = fmaxf(dot(dir, normal), 0.0f);
    if (mixed_pdf > 1e-6f && cos_theta > 0.0f) {
        weight = cos_theta / (M_PI * mixed_pdf);
        weight = fmaxf(0.01f, fminf(weight, 4.0f));  // Clamp to prevent fireflies
    } else {
        weight = 0.0f;
    }
    return dir;
}

#endif // SAMPLING_UTILS_H
