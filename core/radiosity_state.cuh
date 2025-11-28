#ifndef RADIOSITY_STATE_CUH
#define RADIOSITY_STATE_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>

#include "../primitive.h"
#include "../bvh.h"
#include "cuda_utils.cuh"

// ============================================================================
// RADIOSITY STATE MANAGEMENT
// ============================================================================

/**
 * @brief Manages radiosity computation state
 */
struct RadiosityState {
    // Device-side primitives with radiosity data
    Primitive* d_radiosity_primitives;
    
    // Form factors matrix (num_prims x num_prims)
    float* d_form_factors;
    
    // Random states for Monte Carlo integration
    curandState* d_rand_states;
    
    // Solver parameters
    int num_iterations;
    int mc_samples;
    bool use_monte_carlo;
    
    // State flags
    bool is_calculated;
    bool is_converged;
    
    // Statistics
    float last_solve_time_ms;
    float convergence_threshold;
    
    /**
     * @brief Constructor with default initialization
     */
    RadiosityState() 
        : d_radiosity_primitives(nullptr)
        , d_form_factors(nullptr)
        , d_rand_states(nullptr)
        , num_iterations(10)
        , mc_samples(64)
        , use_monte_carlo(true)
        , is_calculated(false)
        , is_converged(false)
        , last_solve_time_ms(0.0f)
        , convergence_threshold(1e-4f)
    {}
    
    /**
     * @brief Run radiosity solver
     * @param h_primitives Host primitives array
     * @param num_prims Number of primitives
     * @param d_bvh_nodes Device BVH nodes
     * @param d_bvh_indices Device BVH indices
     */
    void runSolver(Primitive* h_primitives, int num_prims, 
                   BVHNode* d_bvh_nodes, int* d_bvh_indices);
    
    /**
     * @brief Run a single radiosity iteration
     * @param num_prims Number of primitives
     */
    void runIteration(int num_prims);
    
    /**
     * @brief Calculate form factors
     * @param h_primitives Host primitives
     * @param num_prims Number of primitives
     * @param d_bvh_nodes Device BVH nodes
     * @param d_bvh_indices Device BVH indices
     */
    void calculateFormFactors(Primitive* h_primitives, int num_prims,
                              BVHNode* d_bvh_nodes, int* d_bvh_indices);
    
    /**
     * @brief Check if solver has converged
     * @return true if converged
     */
    bool hasConverged() const { return is_converged; }
    
    /**
     * @brief Reset solver state
     */
    void reset();
    
    /**
     * @brief Cleanup all resources
     */
    void cleanup();
    
    /**
     * @brief Get last solve time
     * @return Solve time in milliseconds
     */
    float getLastSolveTime() const { return last_solve_time_ms; }
};

#endif // RADIOSITY_STATE_CUH
